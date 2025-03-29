# from cosmos_predict1 import instantiate
import sys
sys.path.append("./cosmos_predict1/")
from cosmos_predict1.tokenizer import VideoLossConfig
from cosmos_predict1.tokenizer import FlowLoss
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision.models.optical_flow as optical_flow
IMAGE_KEY = "images"
VIDEO_KEY = "video"
RECON_KEY = "reconstructions"
LATENT_KEY = "latent"
INPUT_KEY = "INPUT"
MASK_KEY = "loss_mask"

_SPATIAL_ALIGN = 16
class WeightScheduler(torch.nn.Module):
    def __init__(self, boundaries, values):
        super().__init__()
        self.boundaries = list(boundaries)
        self.values = list(values)

    def forward(self, iteration):
        for boundary, value in zip(self.boundaries, self.values):
            if iteration < boundary:
                return value
        return self.values[-1]
    
CONFIG_LOSS = {
    "PERCEPTUAL": 
    {
        "lpips_boundaries":[500000],
        "lpips_values": [0.1, 0.073],
        # Layer weights for linearly combining the multi-layer vgg-based losses.
        "layer_weights": [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5],
        # Gram loss, whether to turn on, and what weights to use.
        "gram_enabled": True,
        "gram_boundaries": [500000],
        "gram_values": [0.0, 0.062],
        # Corr loss, whether to turn on, and what weights to use.
        "corr_enabled": False,
        "corr_boundaries": [0],
        "corr_values": [0.0],
        # In the example training memory usage dropped from 64.03 GiB to 60.54 GiB
        # with checkpointing enabled for this loss for about 3.2% slowdown.
        # With checkpointing this and PerceptualLoss memory usage dropped
        # from 64.03 GiB to 52.94 GiB for about 18% slowdown
        # more details in MR:949
        "checkpoint_activations":False
    },
    "FLOW_LOSS": {
    # Flow loss and its weight schedule.
        'boundaries' : [250000],
        'values' : [0.0, 0.01],
        'scale' : 2,
    # Flow loss depends on RAFT, as such it requires a specific dtype.
        'dtype' : "bfloat16",
    # In the example training memory usage dropped from 28GB to 23GB
    # with checkpointing enabled for this loss
    # With checkpointing this and PerceptualLoss memory usage dropped
    # from 64.03 GiB to 52.94 GiB for about 18% slowdown
    # more details in MR:949
        'checkpoint_activations' : False,
        'enabled' : False,
    }
}


class FlowLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.schedule = WeightScheduler(CONFIG_LOSS["FLOW_LOSS"]["boundaries"], CONFIG_LOSS["FLOW_LOSS"]["values"])
        self.scale = CONFIG_LOSS["FLOW_LOSS"]["scale"]
        self.dtype = getattr(torch, CONFIG_LOSS["FLOW_LOSS"]["dtype"])
        self.checkpoint_activations = CONFIG_LOSS["FLOW_LOSS"]["checkpoint_activations"]
        self.enabled = CONFIG_LOSS["FLOW_LOSS"]["enabled"]
        current_device = torch.device("cuda")
        print(current_device)
        def make_coords_grid(
            batch_size: int, h: int, w: int, device: torch.device = current_device, dtype: torch.dtype = self.dtype
        ):
            # Original: def make_coords_grid(batch_size: int, h: int, w: int, device: str = "cpu"):
            device = torch.device(device)
            coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            coords = torch.stack(coords[::-1], dim=0).to(dtype)
            # Original: coords = torch.stack(coords[::-1], dim=0).float()
            return coords[None].repeat(batch_size, 1, 1, 1)

        # We also need to specify output dtype of torch.linspace() in index_pyramid()
        # method of CorrBlock, otherwise it uses default fp32 dtype as output.
        # Additionally I've changed that function to run on GPU instead of CPU, which results in
        # less graph breaks when torch.compile() is used
        # This function is copied from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py#L394
        # commit: b06ea39d5f0adbe949d08257837bda912339e415
        # current_device = torch.device(torch.cuda.current_device())
        def index_pyramid(
            self, centroids_coords, dtype: torch.dtype = self.dtype, device: torch.device = current_device
        ):
            # Original: def index_pyramid(self, centroids_coords):
            """Return correlation features by indexing from the pyramid."""
            neighborhood_side_len = 2 * self.radius + 1  # see note in __init__ about out_channels
            di = torch.linspace(-self.radius, self.radius, neighborhood_side_len, dtype=dtype, device=device)
            dj = torch.linspace(-self.radius, self.radius, neighborhood_side_len, dtype=dtype, device=device)
            # Original: di = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
            # Original: dj = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
            delta = torch.stack(torch.meshgrid(di, dj, indexing="ij"), dim=-1).to(centroids_coords.device)
            delta = delta.view(1, neighborhood_side_len, neighborhood_side_len, 2)

            batch_size, _, h, w = centroids_coords.shape  # _ = 2
            centroids_coords = centroids_coords.permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 2)

            indexed_pyramid = []
            for corr_volume in self.corr_pyramid:
                sampling_coords = centroids_coords + delta  # end shape is (batch_size * h * w, side_len, side_len, 2)
                indexed_corr_volume = optical_flow.raft.grid_sample(
                    corr_volume, sampling_coords, align_corners=True, mode="bilinear"
                ).view(batch_size, h, w, -1)
                indexed_pyramid.append(indexed_corr_volume)
                centroids_coords = centroids_coords / 2

            corr_features = torch.cat(indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()

            expected_output_shape = (batch_size, self.out_channels, h, w)
            if corr_features.shape != expected_output_shape:
                raise ValueError(
                    f"Output shape of index pyramid is incorrect. Should be {expected_output_shape}, got {corr_features.shape}"
                )

            return corr_features

        optical_flow.raft.make_coords_grid = make_coords_grid
        optical_flow.raft.CorrBlock.index_pyramid = index_pyramid

        flow_model = optical_flow.raft_large(pretrained=True, progress=False)
        flow_model.requires_grad_(False)
        flow_model.eval()
        flow_model = flow_model.to(self.dtype)

        self.flow_model = flow_model

    def _run_model(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Runs flow_model in the forward mode on explicit dtype=float32.

        Args:
            input1: First video frames batch, layout (T, C, H, W), bfloat16.
            input2: Next video frames batch, layout (T, C, H, W), bfloat16.

        Returns:
            Forward optical flow, (T, 2, H, W), bfloat16.
        """
        input_dtype = input1.dtype
        flow_output = self.flow_model.to(self.dtype)(input1.to(self.dtype), input2.to(self.dtype))[-1]
        return flow_output.to(input_dtype)

    def _run_model_fwd(self, input_video: torch.Tensor) -> torch.Tensor:
        """Runs foward flow on a batch of videos, one batch at a time.
        Args:
            input_video: The input batch of videos, layout (B, T, C, H, W).

        Returns:
            Forward optical flow, layout (B, 2, T-1, H, W).
        """
        output_list = list()
        for fwd_input_frames in input_video:
            fwd_input_frames = fwd_input_frames.transpose(1, 0)
            fwd_flow_output = self._run_model(fwd_input_frames[:-1], fwd_input_frames[1:])
            output_list.append(fwd_flow_output.transpose(1, 0))
        return torch.stack(output_list, dim=0)

    def _bidirectional_flow(self, input_video: torch.Tensor) -> torch.Tensor:
        """The bidirectional optical flow on a batch of videos.

        The forward and backward flows are averaged to get the bidirectional flow.
        To reduce memory pressure, the input video is scaled down by a factor of `self.scale`,
        and rescaled back to match other pixel-wise losses.

        Args:
            input_video: The input batch of videos, layout (B, T, C, H, W).

        Returns:
            Biderectinoal flow, layout (B, 2, T-1, H, W).
        """
        # scale down the input video to reduce memory pressure.
        t, h, w = input_video.shape[-3:]
        input_video_scaled = F.interpolate(input_video, (t, h // self.scale, w // self.scale), mode="trilinear")

        # forward flow.
        if self.checkpoint_activations:
            fwd_flow_output = checkpoint.checkpoint(self._run_model_fwd, input_video_scaled, use_reentrant=False)
        else:
            fwd_flow_output = self._run_model_fwd(input_video_scaled)

        # backward flow.
        input_video_scaled = input_video_scaled.flip([2])
        if self.checkpoint_activations:
            bwd_flow_output = checkpoint.checkpoint(self._run_model_fwd, input_video_scaled, use_reentrant=False)
        else:
            bwd_flow_output = self._run_model_fwd(input_video_scaled)
        bwd_flow_output = bwd_flow_output.flip([2])

        # bidirectional flow, concat fwd and bwd along temporal axis.
        flow_input = torch.cat([fwd_flow_output, bwd_flow_output], dim=2)
        return self.scale * F.interpolate(flow_input, (2 * (t - 1), h, w), mode="trilinear")

    def forward(
        self, inputs: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor], iteration: int
    ) -> dict[str, torch.Tensor]:
        input_images = inputs[INPUT_KEY]
        if input_images.ndim == 4 or input_images.shape[2] == 1:
            return dict()
        if not self.enabled or self.schedule(iteration) == 0.0:
            return dict()

        # Biderectional flow (B, 2, 2*(T-1), H, W)
        flow_input = self._bidirectional_flow(input_images)
        flow_recon = self._bidirectional_flow(output_batch[RECON_KEY])

        # L1 loss on the flow. (B, 1, 2*(T-1), H, W)
        flow_loss = torch.abs(flow_input - flow_recon).mean(dim=1, keepdim=True)

        flow_loss_weighted = self.schedule(iteration) * flow_loss
        if torch.isnan(flow_loss_weighted).any():
            raise ValueError("[FLOW] NaN detected in loss")
        return dict(flow=flow_loss_weighted)

    def torch_compile(self):
        """
        This method invokes torch.compile() on this loss
        """
        self.flow_model = torch.compile(self.flow_model, dynamic=False)

if __name__ == "__main__":
    fl_loss = FlowLoss()