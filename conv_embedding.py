import torch
import torch.nn as nn

class ConvEmbedding(nn.Module):
    def __init__(self, nb_neighbors, kernel_size, dim_emb, dim_input_nodes):
        super().__init__()
        self.nb_neighbors = nb_neighbors
        self.conv = nn.Conv1d(in_channels=dim_input_nodes, out_channels=dim_emb, kernel_size=kernel_size)
        self.W1 = nn.Linear(dim_input_nodes, dim_emb)  # for node x_i
        self.W2 = nn.Linear(dim_emb, dim_emb)  # for convolved node feature hbar_i

    def forward(self, x):
        """
        :param Tensor x: (B, N, 2)
        :return Tensor final_embedding: (B, N, H)
        """
        bsz, seq_len = x.size(0), x.size(1)

        node_embedding = self.W1(x)  # (B, N, H)

        # Make k-NN for each node (B, N, K+1, 2)
        dist_matrix = torch.cdist(x, x)  # (B, N, N)
        # knn_indices = dist_matrix.topk(self.nb_neighbors+1)[1]  # (B, N, K+1) including itself
        knn_indices = dist_matrix.topk(k=seq_len)[1][:,:,-self.nb_neighbors-1:]  # (B, N, K+1) including itself
        embedding_list = []
        for i in range(seq_len):
            idx = knn_indices[:, i, :].unsqueeze(2).repeat(1,1,2)  # (B, K+1, 2)
            knn_coords = x.gather(1, idx)

            knn_coords = knn_coords.permute(0, 2, 1)  # (B, 2, K+1)
            conv_embedding = self.conv(knn_coords)  # (B, H, 1)
            
            conv_embedding = conv_embedding.permute(0, 2, 1)  # (B, 1, H)
            # conv_embedding = self.W2(conv_embedding)  # (B, 1, H)  comment

            embedding_list.append(conv_embedding)
        conv_embedding = torch.cat(embedding_list, dim=1)  # (B, N, H)
        conv_embedding = self.W2(conv_embedding)  # (B, N, H)  namely CEFix

        final_embedding = node_embedding + conv_embedding  # (B, N, H)
        return final_embedding

class ConvSamePadding(nn.Module):
    def __init__(self, dim_input_nodes, dim_emb, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=dim_input_nodes, out_channels=dim_emb, kernel_size=kernel_size, padding='same')
    def forward(self, x):
        """(B, N, 2) --> (B, N, H)"""
        x = x.permute(0,2,1)  # (B, 2, N)
        h = self.conv(x)  # (B, H, N)
        h = h.permute(0, 2, 1)  # (B, N, H)
        return h

class ConvLinear(nn.Module):
    def __init__(self, dim_input_nodes, dim_emb, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=dim_input_nodes, out_channels=dim_emb, kernel_size=kernel_size, padding='same')
        self.W1 = nn.Linear(dim_input_nodes, dim_emb)
        self.W2 = nn.Linear(dim_emb, dim_emb)
    def forward(self, x):
        """(B, N, 2) --> (B, N, H)"""
        node_embedding = self.W1(x)  # (B, N, H)
        x = x.permute(0, 2, 1)  # (B, 2, N)
        conv_embedding = self.conv(x)  # (B, H, N)
        conv_embedding = conv_embedding.permute(0, 2, 1)  # (B, N, H)
        final_embedding = node_embedding + conv_embedding
        return final_embedding

class ConvEmbeddingXY(nn.Module):
    def __init__(self, nb_neighbors, kernel_size, dim_emb, dim_input_nodes):
        super().__init__()
        self.nb_neighbors = nb_neighbors
        self.conv_x = nn.Conv1d(in_channels=dim_input_nodes, out_channels=dim_emb, kernel_size=kernel_size)
        self.conv_y = nn.Conv1d(in_channels=dim_input_nodes, out_channels=dim_emb, kernel_size=kernel_size)
        self.W1 = nn.Linear(dim_input_nodes, dim_emb)  # for node x_i
        self.W2 = nn.Linear(dim_emb, dim_emb)  # for convolved node feature hbar_i
    
    def _sort_by_xy(self, coord):
        """
        :param Tensor coord: (B, K, 2)
        """
        toB = torch.arange(coord.size(0))
        indices = coord[:, :, 0].argsort()  # sort by x coordinate (B, K)
        coord = coord[toB.unsqueeze(1), indices]  # (B, K, 2)
        return coord
    
    def _sort_by_yx(self, coord):
        toB = torch.arange(coord.size(0))
        indices = coord[:, :, 1].argsort()  # sort by y coordinate
        coord = coord[toB.unsqueeze(1), indices]
        return coord

    def forward(self, x):
        """
        :param Tensor x: (B, N, 2)
        :return Tensor final_embedding: (B, N, H)
        """
        bsz, seq_len = x.size(0), x.size(1)

        node_embedding = self.W1(x)  # (B, N, H)

        # Make k-NN for each node (B, N, K+1, 2)
        dist_matrix = torch.cdist(x, x)  # (B, N, N)
        knn_indices = dist_matrix.topk(k=seq_len)[1][:,:,-self.nb_neighbors-1:]  # (B, N, K+1) including itself

        embedding_list = []
        for i in range(seq_len):
            idx = knn_indices[:, i, :].unsqueeze(2).repeat(1,1,2)  # (B, K+1, 2)
            knn_coords = x.gather(1, idx)

            knn_coords_x = self._sort_by_xy(knn_coords)  # (B, K+1, 2)
            knn_coords_y = self._sort_by_yx(knn_coords)  # (B, K+1, 2)

            knn_coords_x = knn_coords_x.permute(0, 2, 1)  # (B, 2, K+1)
            conv_embedding_x = self.conv_x(knn_coords_x)  # (B, H, 1)
            
            knn_coords_y = knn_coords_y.permute(0, 2, 1)  # (B, 2, K+1)
            conv_embedding_y = self.conv_y(knn_coords_y)  # (B, H, 1)

            conv_embedding = conv_embedding_x + conv_embedding_y  # (B, H, 1)

            conv_embedding = conv_embedding.permute(0, 2, 1)  # (B, 1, H)
            # conv_embedding = self.W2(conv_embedding)  # (B, 1, H)

            embedding_list.append(conv_embedding)
        conv_embedding = torch.cat(embedding_list, dim=1)  # (B, N, H)
        conv_embedding = self.W2(conv_embedding)  # (B, N, H)  namely CEFix

        final_embedding = node_embedding + conv_embedding  # (B, N, H)
        return final_embedding
     
if __name__ == '__main__':
    x = torch.rand(1, 10, 2)
    sorter = 'sum_xy'
    # emb = ConvEmbedding(5, 6, 128, 2, sorter)
    emb = ConvEmbeddingXY(5, 6, 128, 2)
    emb(x)
