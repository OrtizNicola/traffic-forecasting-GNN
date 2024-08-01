import torch
import torch.nn as nn 
from torch_geometric.nn import GATv2Conv, GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GATv2_GRU_Model(nn.Module):
    def __init__(self, node_feature_size=1, hidden_dim_gat=2, hidden_dim_gru=4, gru_layers=2, output_size=1, num_nodes=207, graf_layers=1, heads=1):
        super(GATv2_GRU_Model, self).__init__()
        self.gat_conv = GATv2Conv(node_feature_size, hidden_dim_gat, heads, concat=False)
        self.gat_conv2 = GATv2Conv(hidden_dim_gat, hidden_dim_gat, heads, concat=False)
        self.gru = nn.GRU(hidden_dim_gat * num_nodes, hidden_dim_gru * num_nodes, gru_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim_gru * num_nodes, output_size * num_nodes)
        self.num_nodes = num_nodes
        self.graf_layers = graf_layers

    def forward(self, graph_sequence, edge_index):
        batch_size, seq_len, num_nodes, node_feature_size = graph_sequence.size()

        # Preparar las secuencias de grafos para GATConv
        graph_sequence = graph_sequence.view(batch_size * seq_len, num_nodes, node_feature_size)
        x = graph_sequence.view(-1, node_feature_size)  # (batch_size * seq_len * num_nodes) x node_feature_size

        # Ajustar edge_index para batch processing
        edge_index_batch = edge_index.repeat(1, batch_size * seq_len)
        offset = torch.arange(0, batch_size * seq_len * num_nodes, step=num_nodes, dtype=torch.long).repeat_interleave(edge_index.size(1))
        offset = offset.to(device)
        edge_index_batch = edge_index_batch + offset

        # Procesar todos los grafos en la secuencia de una vez con GATConv
        gat_output = self.gat_conv(x, edge_index_batch) # aplicar primera capa de red de grafos

        if self.graf_layers == 2:
            gat_output = self.gat_conv2(gat_output, edge_index_batch) # segunda capa de la red de grafos

        gat_output = gat_output.view(batch_size, seq_len, num_nodes, -1) # -1 = hidden_dim para las features de cada nodo

        # Pasar las salidas de GATConv a GRU
        gat_output = gat_output.view(batch_size, seq_len, -1)  # batch_size x seq_len x (num_nodes * hidden_dim)
        gru_out, _ = self.gru(gat_output)

        # Predecir el siguiente estado del grafo
        gru_out = gru_out[:, -1, :]  # batch_size x hidden_dim
        out = self.fc(gru_out)
        out = out.view(batch_size, num_nodes, -1)  # batch_size x num_nodes x output_size

        return out
    
class GATv2_LSTM_Model(nn.Module):
    def __init__(self, node_feature_size=1, hidden_dim_gat=2, hidden_dim_lstm=4, lstm_layers=2, output_size=1, num_nodes=207, graf_layers=1, heads=1):
        super(GATv2_LSTM_Model, self).__init__()
        self.gat_conv = GATv2Conv(node_feature_size, hidden_dim_gat, heads, concat=False)
        self.gat_conv2 = GATv2Conv(hidden_dim_gat, hidden_dim_gat, heads, concat=False)
        self.lstm = nn.LSTM(hidden_dim_gat * num_nodes, hidden_dim_lstm * num_nodes, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim_lstm * num_nodes, output_size * num_nodes)
        self.num_nodes = num_nodes
        self.graf_layers = graf_layers

    def forward(self, graph_sequence, edge_index):
        batch_size, seq_len, num_nodes, node_feature_size = graph_sequence.size()

        # Preparar las secuencias de grafos para GATConv
        graph_sequence = graph_sequence.view(batch_size * seq_len, num_nodes, node_feature_size)
        x = graph_sequence.view(-1, node_feature_size)  # (batch_size * seq_len * num_nodes) x node_feature_size

        # Ajustar edge_index para batch processing
        edge_index_batch = edge_index.repeat(1, batch_size * seq_len)
        offset = torch.arange(0, batch_size * seq_len * num_nodes, step=num_nodes, dtype=torch.long).repeat_interleave(edge_index.size(1))
        offset = offset.to(device)
        edge_index_batch = edge_index_batch + offset

        # Procesar todos los grafos en la secuencia de una vez con GATConv
        gat_output = self.gat_conv(x, edge_index_batch) # aplicar primera capa de red de grafos

        if self.graf_layers == 2:
            gat_output = self.gat_conv2(gat_output, edge_index_batch) # segunda capa de la red de grafos

        gat_output = gat_output.view(batch_size, seq_len, num_nodes, -1) # -1 = hidden_dim para las features de cada nodo

        # Pasar las salidas de GATConv a LSTM
        gat_output = gat_output.view(batch_size, seq_len, -1)  # batch_size x seq_len x (num_nodes * hidden_dim)
        lstm_out, _ = self.lstm(gat_output)

        # Predecir el siguiente estado del grafo
        lstm_out = lstm_out[:, -1, :]  # batch_size x hidden_dim
        out = self.fc(lstm_out)
        out = out.view(batch_size, num_nodes, -1)  # batch_size x num_nodes x output_size

        return out
    
class GCN_GRU_Model(nn.Module):
    def __init__(self, node_feature_size=1, hidden_dim_gcn=2, hidden_dim_gru=4, gru_layers=2, output_size=1, num_nodes=207, graf_layers=1):
        super(GCN_GRU_Model, self).__init__()
        self.gcn_conv = GCNConv(node_feature_size, hidden_dim_gcn)
        self.gcn_conv2 = GCNConv(hidden_dim_gcn, hidden_dim_gcn)
        self.gru = nn.GRU(hidden_dim_gcn * num_nodes, hidden_dim_gru * num_nodes, gru_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim_gru * num_nodes, output_size * num_nodes)
        self.num_nodes = num_nodes
        self.graf_layers = graf_layers

    def forward(self, graph_sequence, edge_index):
        batch_size, seq_len, num_nodes, node_feature_size = graph_sequence.size()

        # Preparar las secuencias de grafos para GCNConv
        graph_sequence = graph_sequence.view(batch_size * seq_len, num_nodes, node_feature_size)
        x = graph_sequence.view(-1, node_feature_size)  # (batch_size * seq_len * num_nodes) x node_feature_size

        # Ajustar edge_index para batch processing
        edge_index_batch = edge_index.repeat(1, batch_size * seq_len)
        offset = torch.arange(0, batch_size * seq_len * num_nodes, step=num_nodes, dtype=torch.long).repeat_interleave(edge_index.size(1))
        offset = offset.to(device)
        edge_index_batch = edge_index_batch + offset

        # Procesar todos los grafos en la secuencia de una vez con GCNConv
        gcn_output = self.gcn_conv(x, edge_index_batch) # primera capa del gcn

        if self.graf_layers == 2:
            gcn_output = self.gcn_conv2(gcn_output, edge_index_batch) # segudna capa del gcn

        gcn_output = gcn_output.view(batch_size, seq_len, num_nodes, -1) # -1 = hidden_dim para las features de cada nodo

        # Pasar las salidas de GCNConv a GRU
        gcn_output = gcn_output.view(batch_size, seq_len, -1)  # batch_size x seq_len x (num_nodes * hidden_dim)
        gru_out, _ = self.gru(gcn_output)

        # Predecir el siguiente estado del grafo
        gru_out = gru_out[:, -1, :]  # batch_size x hidden_dim
        out = self.fc(gru_out)
        out = out.view(batch_size, num_nodes, -1)  # batch_size x num_nodes x output_size

        return out
    
class GCN_LSTM_Model(nn.Module):
    def __init__(self, node_feature_size=1, hidden_dim_gcn=2, hidden_dim_lstm=4, lstm_layers=2, output_size=1, num_nodes=207, graf_layers=1):
        super(GCN_LSTM_Model, self).__init__()
        self.gcn_conv = GCNConv(node_feature_size, hidden_dim_gcn)
        self.gcn_conv2 = GCNConv(hidden_dim_gcn, hidden_dim_gcn)
        self.lstm = nn.LSTM(hidden_dim_gcn * num_nodes, hidden_dim_lstm * num_nodes, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim_lstm * num_nodes, output_size * num_nodes)
        self.num_nodes = num_nodes
        self.graf_layers = graf_layers

    def forward(self, graph_sequence, edge_index):
        batch_size, seq_len, num_nodes, node_feature_size = graph_sequence.size()

        # Preparar las secuencias de grafos para GCNConv
        graph_sequence = graph_sequence.view(batch_size * seq_len, num_nodes, node_feature_size)
        x = graph_sequence.view(-1, node_feature_size)  # (batch_size * seq_len * num_nodes) x node_feature_size

        # Ajustar edge_index para batch processing
        edge_index_batch = edge_index.repeat(1, batch_size * seq_len)
        offset = torch.arange(0, batch_size * seq_len * num_nodes, step=num_nodes, dtype=torch.long).repeat_interleave(edge_index.size(1))
        offset = offset.to(device)
        edge_index_batch = edge_index_batch + offset

        # Procesar todos los grafos en la secuencia de una vez con GCNConv
        gcn_output = self.gcn_conv(x, edge_index_batch) # primera capa del gcn

        if self.graf_layers == 2:
            gcn_output = self.gcn_conv2(gcn_output, edge_index_batch) # segunda capa del gcn

        gcn_output = gcn_output.view(batch_size, seq_len, num_nodes, -1) # -1 = hidden_dim para las features de cada nodo

        # Pasar las salidas de GCNConv a LSTM
        gcn_output = gcn_output.view(batch_size, seq_len, -1)  # batch_size x seq_len x (num_nodes * hidden_dim)
        lstm_out, _ = self.lstm(gcn_output)

        # Predecir el siguiente estado del grafo
        lstm_out = lstm_out[:, -1, :]  # batch_size x hidden_dim
        out = self.fc(lstm_out)
        out = out.view(batch_size, num_nodes, -1)  # batch_size x num_nodes x output_size

        return out
