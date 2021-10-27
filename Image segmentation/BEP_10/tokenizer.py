from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Customize pre-tokenization and decoding
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# And then train
trainer = trainers.BpeTrainer(vocab_size=20000, min_frequency=2)
tokenizer.train([
	"short_text.txt"
], trainer=trainer)
# And Save it
tokenizer.save("byte-level-bpe.tokenizer.json", pretty=True)

tokenizer = Tokenizer.from_file("byte-level-bpe.tokenizer.json")

encoded = tokenizer.encode("I can feel the magic, can you?")
print(encoded.tokens)