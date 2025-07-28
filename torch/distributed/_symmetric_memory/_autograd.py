from typing import Optional

import torch
import torch.distributed._symmetric_memory as symm_mem


class AllToAllVDev2d(torch.autograd.Function):
    """
    Autograd function for `all_to_all_vdev_2d`
    Usage:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Initialize
        >>> AllToAllVDev2d.init(max_output_len)
        >>> # Forward
        >>> output, out_splits_offsets = AllToAllVDev2d.apply(
        ...     input, input_splits, group_name, major_align
        ... )
        >>> # Backward
        >>> s = output.sum()
        >>> s.backward()
        >>> print(input.grad)
    """

    # Maximum output length (need to be set before use of AllToAllVDev2d)
    max_output_len: Optional[int] = None
    # A symmetric memory holding the grad_output during backward
    grad_output_buf: Optional[torch.Tensor] = None

    @staticmethod
    def init(
        max_output_len: int,
    ) -> None:
        """
        Init function for `AllToAllVDev2d`. Must be called before use.
        Args:
            `max_output_len`: maximum output length, must be larger than the actual output length.
        """
        AllToAllVDev2d.max_output_len = max_output_len

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx,
        input: torch.Tensor,
        input_splits: torch.Tensor,
        group_name: str,
        major_align: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Functionality is the same as `all_to_all_vdev_2d` but with `output` and
        `out_splits_offsets` returned instead of modified in place.
        """
        if AllToAllVDev2d.max_output_len is None:
            raise RuntimeError(
                "Please set max output length via `AllToAllVDev2d.init(max_output_len)`"
            )

        # Allocate output buffer
        output = symm_mem.empty(
            (AllToAllVDev2d.max_output_len, *input.shape[1:]),
            dtype=input.dtype,
            device=input.device,
        )
        # Allocate output splits tensor
        out_splits_offsets = symm_mem.empty(
            (2, *input_splits.shape),
            dtype=input_splits.dtype,
            device=input_splits.device,
        )

        # Shuffle input to output
        torch.ops.symm_mem.all_to_all_vdev_2d(
            input, output, input_splits, out_splits_offsets, group_name, major_align
        )

        # Output splits in forward is the input splits in backward
        ctx.save_for_backward(out_splits_offsets)
        ctx.group_name = group_name
        ctx.input_shape = input.shape
        return output, out_splits_offsets

    @staticmethod
    def backward(  # type: ignore[no-untyped-def]
        ctx,
        grad_output: torch.Tensor,
        grad_out_splits_offsets_unused,
    ) -> tuple[torch.Tensor, None, None, None]:
        """
        Backward pass of `all_to_all_vdev_2d` is `all_to_all_vdev_2d_offset`.

        Args:
            `grad_output`: gradients of output passed back from the downstream.
            `grad_out_splits_offsets_unused`: gradients of `out_splits_offsets`, this is usually None, unused.

        Returns:
            `grad_input`: gradients of input.
        """

        # Initialize grad_output buffer (one time only)
        if AllToAllVDev2d.grad_output_buf is None:
            assert AllToAllVDev2d.max_output_len is not None, "`max_output_len` not set"
            AllToAllVDev2d.grad_output_buf = symm_mem.empty(
                AllToAllVDev2d.max_output_len,
                *grad_output.shape[1:],
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

        # TODO: is there a way to tell autograd to feed grad_output directly to
        # our symm_mem buffer?
        AllToAllVDev2d.grad_output_buf.narrow(0, 0, grad_output.shape[0]).copy_(
            grad_output
        )

        # Splits info
        # Splits/offsets of grad_out is the same as out splits/offsets in forward
        (grad_out_splits_offsets,) = ctx.saved_tensors
        grad_in_splits_offsets = symm_mem.empty(
            *grad_out_splits_offsets.shape,
            dtype=grad_out_splits_offsets.dtype,
            device=grad_out_splits_offsets.device,
        )
        grad_input = symm_mem.empty(
            *ctx.input_shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )

        # Shuffle gradients back to the input
        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            AllToAllVDev2d.grad_output_buf,
            grad_input,
            grad_out_splits_offsets,
            grad_in_splits_offsets,
            group_name=ctx.group_name,
        )
        return grad_input, None, None, None


class AllToAllVDev2dOffset(torch.autograd.Function):
    """
    Autograd function for `all_to_all_vdev_2d_offset`
    Usage:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Initialize
        >>> AllToAllVDev2dOffset.init(max_output_len)
        >>> # Forward
        >>> output, out_splits_offsets = AllToAllVDev2dOffset.apply(
        ...     input, input_splits, group_name, major_align
        ... )
        >>> # Backward
        >>> s = output.sum()
        >>> s.backward()
        >>> print(input.grad)
    """

    # Maximum output length (need to be set before use of AllToAllVDev2dOffset)
    max_output_len: Optional[int] = None
    # Alignment of the input offsets
    alignment: Optional[int] = None
    # A symmetric memory holding the grad_output during backward
    grad_output_buf: Optional[torch.Tensor] = None

    @staticmethod
    def init(
        max_output_len: int,
        alignment: int,
    ) -> None:
        """
        Init function for `AllToAllVDev2dOffset`. Must be called before use.
        Args:
            `max_output_len`: maximum output length, must be larger than the actual output length.
        """
        AllToAllVDev2dOffset.max_output_len = max_output_len
        AllToAllVDev2dOffset.alignment = alignment

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx,
        input: torch.Tensor,
        in_splits_offsets: torch.Tensor,
        group_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Functionality is the same as `all_to_all_vdev_2d_offset` but with `output` and
        `out_splits_offsets` returned instead of modified in place.
        """
        if AllToAllVDev2dOffset.max_output_len is None:
            raise RuntimeError(
                "Please set max output length via `AllToAllVDev2dOffset.init(max_output_len)`"
            )

        # Allocate output buffer
        output = symm_mem.empty(
            (AllToAllVDev2dOffset.max_output_len, *input.shape[1:]),
            dtype=input.dtype,
            device=input.device,
        )
        # Allocate output splits tensor
        out_splits_offsets = symm_mem.empty(
            in_splits_offsets.shape,
            dtype=in_splits_offsets.dtype,
            device=in_splits_offsets.device,
        )

        # Shuffle input to output
        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            input, output, in_splits_offsets, out_splits_offsets, group_name
        )

        # Output splits in forward is the input splits in backward
        ctx.save_for_backward(out_splits_offsets)
        ctx.group_name = group_name
        ctx.input_shape = input.shape
        return output, out_splits_offsets

    @staticmethod
    def backward(  # type: ignore[no-untyped-def]
        ctx,
        grad_output: torch.Tensor,
        grad_out_splits_offsets_unused,
    ) -> tuple[torch.Tensor, None, None, None]:
        """
        Backward pass of `all_to_all_vdev_2d_offset` is `all_to_all_vdev_2d`.

        Args:
            `grad_output`: gradients of output passed back from the downstream.
            `grad_out_splits_offsets_unused`: gradients of `out_splits_offsets`, this is usually None, unused.

        Returns:
            `grad_input`: gradients of input.
        """

        # Initialize grad_output buffer (one time only)
        if AllToAllVDev2dOffset.grad_output_buf is None:
            assert AllToAllVDev2dOffset.max_output_len is not None, (
                "`max_output_len` not set"
            )
            AllToAllVDev2dOffset.grad_output_buf = symm_mem.empty(
                AllToAllVDev2dOffset.max_output_len,
                *grad_output.shape[1:],
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

        # TODO: is there a way to tell autograd to feed grad_output directly to
        # our symm_mem buffer?
        AllToAllVDev2dOffset.grad_output_buf.narrow(0, 0, grad_output.shape[0]).copy_(
            grad_output
        )

        # Splits info
        # Splits/offsets of grad_out is the same as out splits/offsets in forward
        (grad_out_splits_offsets,) = ctx.saved_tensors
        grad_out_splits = grad_out_splits_offsets[0]
        grad_in_splits_offsets = symm_mem.empty(
            *grad_out_splits_offsets.shape,
            dtype=grad_out_splits_offsets.dtype,
            device=grad_out_splits_offsets.device,
        )
        grad_input = symm_mem.empty(
            *ctx.input_shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )

        # Shuffle gradients back to the input
        # TODO: create an op that takes both in_splits_offsets and
        # out_splits_offsets, instead of taking alignment
        torch.ops.symm_mem.all_to_all_vdev_2d(
            AllToAllVDev2dOffset.grad_output_buf,
            grad_input,
            grad_out_splits,
            grad_in_splits_offsets,
            ctx.group_name,
            AllToAllVDev2dOffset.alignment,
        )

        return grad_input, None, None, None
