from FlexiEmbed import *


model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)


img_path = 'image.jpg'
image = Image.open(img_path)

square_size = 224
patch_size = 16

squares = auto_overlap_crop(image, square_size)
embeddings = get_embeddings_from_squares(squares, model, feature_extractor)
final_embeddings = reshape_embeddings(embeddings)
trimmed_final_embeddings=trim_edges_embeddings(image,final_embeddings,square_size,patch_size)

print("Size of the final embedding array:", trimmed_final_embeddings.shape)

