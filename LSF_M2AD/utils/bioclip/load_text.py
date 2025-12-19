from open_clip.model import CLIPTextCfg, _build_text_tower
import torch
from open_clip.tokenizer import tokenize
import os

class TextEncoder:
    def __init__(self, model_path=None, device=None):
        """
        Initialize the Text Encoder for BioCLIP.
        
        Args:
            model_path (str): Path to the pre-trained weights. If None, looks for 'text_encoder.pth' in the script directory.
            device (torch.device): Device to load the model onto.
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "text_encoder.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"\n[Error] BioCLIP weights not found at: {model_path}\n"
                f"Please download 'text_encoder.pth' from the following link and place it in the directory:\n"
                f"Download Link: https://drive.google.com/file/d/1vTUXp3WFamr6okZVPNb_JLzIw192v8bO/view?usp=sharing"
            )

        # Determine device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load saved text encoder state dictionary
        self.text_encoder_state_dict = torch.load(model_path, map_location=self.device)
        
        # Get vocabulary size
        vocab_size = self.text_encoder_state_dict.pop('vocab_size', 49408)  # Default to standard CLIP vocab size
        
        # Configure text encoder parameters (Standard ViT-B-16 configuration)
        self.text_cfg = CLIPTextCfg(
            context_length=77,
            vocab_size=vocab_size,
            width=512,
            heads=8,
            layers=12
        )
        
        # Create text encoder model
        self.embed_dim = 512  # Standard embedding dimension for ViT-B-16
        self.model = self._initialize_model()
        
        # Freeze parameters
        self.lock_model()
    
    def lock_model(self):
        """Freeze model parameters to prevent updates during training."""
        for param in self.model.parameters():
            param.requires_grad = False
        print("TextEncoder parameters locked. Training will not update these weights.")
    
    def _initialize_model(self):
        """Initialize and load the model architecture with saved weights."""
        model = _build_text_tower(self.embed_dim, self.text_cfg)
        
        # Convert state dict to match standalone structure if necessary
        new_state_dict = {}
        for name, param in self.text_encoder_state_dict.items():
            new_name = name
            if name.startswith('transformer.'):
                new_name = 'transformer.' + name[len('transformer.'):]
            elif name.startswith('token_embedding.'):
                new_name = 'token_embedding.' + name[len('token_embedding.'):]
            new_state_dict[new_name] = param
        
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(self.device)
        return model
    
    def encode(self, text, normalize=True):
        """
        Encode text into feature vectors.
        
        Args:
            text (str or list): Text or list of strings to encode.
            normalize (bool): Whether to L2 normalize the features.
            
        Returns:
            torch.Tensor: Encoded text features.
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize and move to device
        tokens = tokenize(text).to(self.device)
        
        # Encode features
        with torch.no_grad():
            features = self.model(tokens)
        
        # Optional L2 normalization
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features
    
    def __call__(self, tokens):
        """Direct call for compatibility with older codebases."""
        if hasattr(tokens, 'device') and tokens.device != self.device:
            tokens = tokens.to(self.device)
        
        with torch.no_grad():
            return self.model(tokens)

# Create default instance
try:
    text_encoder = TextEncoder()
except FileNotFoundError as e:
    print(e)
    text_encoder = None

# Simple testing
if __name__ == "__main__":
    if text_encoder:
        test_text = "a test Alzheimer's report snippet"
        features = text_encoder.encode(test_text)
        print(f"Feature shape: {features.shape}")
    else:
        print("TextEncoder initialization failed due to missing weights.")
