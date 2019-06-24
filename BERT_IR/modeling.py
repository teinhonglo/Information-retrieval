import torch

from pytorch_pretrained_bert.modeling import BertModel,BertConfig,BertEmbeddings,BertForSequenceClassification

class BertEmbeddingsForTDT(BertEmbeddings):
    def __init__(self,config):
        super(BertEmbeddingsForTDT, self).__init__(config)
    
    def forward(self, input_ids, token_type_ids=None, input_weight=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        #input_weight_embeddings = self.word_embeddings(input_weight)
       
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = embeddings * input_weight
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModelForTDT(BertModel):
    def __init__(self,config):
        super(BertModelForTDT, self).__init__(config)
        self.embeddings = BertEmbeddingsForTDT(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_weight=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) 
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids, input_weight)
        
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class BertForSequenceClassificationForTDT(BertForSequenceClassification):
    def __init__(self, config,num_labels):
        super(BertForSequenceClassificationForTDT, self).__init__(config, num_labels)
        self.bert = BertModelForTDT(config)
   
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_weight=None,labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, input_weight, output_all_encoded_layers=False)
        print(pooled_output.size())
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


if  __name__ == '__main__':
    '''
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    input_weight = torch.FloatTensor([[[0.222], [0.432], [0.888]], [[0.32], [0.1], [0.005]]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertModelForTDT(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask, input_weight)
    '''
    # Classification
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    input_weight = torch.FloatTensor([[[0.222], [0.432], [0.888]], [[0.32], [0.1], [0.005]]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassificationForTDT.from_pretrained('bert-base-chinese', num_labels=num_labels)#(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask, input_weight)

