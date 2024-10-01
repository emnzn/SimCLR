import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXent(nn.Module):

    """
    Constructs the NTXent loss function for Contrastive Learning.
    """

    def __init__(
        self, 
        temperature: float = 0.5
        ):
        super().__init__()

        self.temperature = temperature

    def _get_target_mask(
        self, 
        batch_size: int
        ) -> torch.Tensor:

        """
        Creates a target mask for a batch of data.

        This mask identifies which parts of the batch should not be compared with themselves.
        It returns a tensor where the diagonal elements are `False` (indicating that an instance should not be compared with itself),
        and the off-diagonal elements are `True` (indicating that other instances in the batch should be compared).

        Parameters
        ----------
        batch_size: int
            The number of instances in the batch, which determines the size of the mask.

        Returns
        -------
        target_mask: torch.Tensor
            A boolean mask of shape (batch_size, batch_size) where `False` represents the 
            identity (self-comparison) and `True` represents valid comparisons.
        """

        mask = torch.eye(batch_size)
        target_mask = torch.logical_not(mask)

        return target_mask

    def forward(
        self, 
        z_i: torch.Tensor, 
        z_j: torch.Tensor
        ) -> torch.Tensor:

        """
        Defines the forward pass of the model.

        Parameters
        ----------
        z_i: torch.Tensor
            The projected representation for the first view of the image.

        z_j: torch.Tensor
            The projected representation for the second view of the image.

        Returns
        -------
        loss: torch.Tensor
            The contrastive loss as defined by NTXent.
        """

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = representations @ representations.T

        batch_size = z_i.shape[0]
        target_mask = self._get_target_mask(2*batch_size).to(representations.device)

        positive_pairs = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
            ], dim=0)
        
        negative_pairs = similarity_matrix * target_mask

        numerator = torch.exp(positive_pairs / self.temperature)
        denominator = torch.sum(torch.exp(negative_pairs / self.temperature), dim=1)

        loss = -torch.log(numerator/denominator)
        loss = torch.mean(loss)

        return loss
