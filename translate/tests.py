import torch
import unittest

import models

class TestTransformerBlock(unittest.TestCase):
    def test_transformer_output_shape(self):
        # Define the model parameters
        num_encoder_layers = 2
        num_decoder_layers = 2
        d_model = 512
        num_heads = 8
        dff = 2048
        input_vocab_size = 1000  # Assume a vocabulary size of 1000 for the input
        target_vocab_size = 1000  # Assume a vocabulary size of 1000 for the target
        max_seq_length = 60
        dropout_rate = 0.1

        # Initialize the Transformer model
        model = models.Transformer(num_encoder_layers, num_decoder_layers, d_model, num_heads,
                            dff, input_vocab_size, target_vocab_size, 
                            pe_input=max_seq_length, pe_target=max_seq_length, 
                            dropout_rate=dropout_rate)

        # Generate dummy input data
        batch_size = 32
        input_seq_length = 45
        target_seq_length = 50
        dummy_input = torch.randint(0, input_vocab_size, (batch_size, input_seq_length))
        dummy_target = torch.randint(0, target_vocab_size, (batch_size, target_seq_length))

        # Dummy masks (in a real scenario, these should be properly computed)
        enc_padding_mask, look_ahead_mask, dec_padding_mask = None, None, None

        # Forward pass
        output = model(dummy_input, dummy_target, enc_padding_mask, look_ahead_mask, dec_padding_mask)

        # Check the output shape
        expected_shape = (batch_size, target_seq_length, target_vocab_size)
        self.assertEqual(output.shape, expected_shape)

if __name__ == "__main__":
    unittest.main()
