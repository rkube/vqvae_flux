using Flux
using Distributions
using Random
using MLDatasets: MNIST
using CairoMakie


"""
    Load MNIST
"""
data_train = MNIST(:train);
X_train, labels_train = data_train[:];
# scale to -0.5:0.5 in order to use tanh
#all_X_train = (all_X_train .- 5f-1) .* 2f0;
X_train = Flux.unsqueeze(X_train, dims=3) |> gpu;

X_test, labels_test = MNIST(:test)[:];
X_test = Flux.unsqueeze(X_test, dims=3) |> gpu;

"""
    Embedding(num_embeddings, embedding_dim)

Transforms a tensor of integers to dense vectors. This layer defines a 
vocabulary as `num_embeddings` vectors or dimension `embedding_dim`.
In transformer language, `num_embeddings` corresponds to the size
of the vocabulary. And `embedding_dim` refers to the dimension of the vectors.

Inspired by https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/

The weights are by default initialized using a uniform distribution over
(-1/num_embeddings, num_embeddings)
"""

struct Embedding{F, T<: AbstractArray{F}}
    embedding::T
end

Flux.@functor Embedding

function Embedding(num_embeddings::Int, embedding_dim::Int, init = Flux.glorot_uniform) 
    weights = Flux.glorot_uniform(num_embeddings, embedding_dim)
    Embedding(weights)
end


"""
    Encoder for MNIST
"""

# Use the channel dimension to quantize over.
# That means, that the final number of channels in encoder_features
# needs to match the 

num_channels = 64

embedding_dim = 64
num_embeddings = 128

batch_size = 5

encoder_features = Chain(
    Conv((5, 5), 1 => num_channels ÷ 4, relu; stride=1), # 24x24
    Conv((5, 5), num_channels ÷ 4 => num_channels ÷ 2, relu; stride=1), # 20x20
    Conv((5, 5), num_channels ÷ 2 => num_channels, relu, stride=2), # 8x8
    Conv((5, 5), num_channels => embedding_dim, relu, stride=2), # 2x2
    x -> permutedims(x, (1, 2, 4, 3)),    # Channels are now at the last dimension
    Flux.flatten # 2x2x64, batch_size
)


# Size is HWCB
X = randn(Float32, 28, 28, 1, batch_size);

# Size is (H' * W' * B) x C
# That is, we have (H' * W' * B) vectors of size C
size(encoder_features(X))

# code <-> flat_input. Flattened output of encoder network
code = encoder_features(X)

# Get an embeddding
e = Embedding(num_embeddings, embedding_dim);

term1 = sum(code.^2, dims=2);
term2 = sum((e.embedding).^2, dims=2)
term3 = code * e.embedding';

# Project the flattened output of the encoder onto the embedding vectors.
distances = sum(code.^2, dims=2) .+ sum((e.embedding).^2, dims=2)' .- 2f0 .* code * e.embedding'

# Find the embedding vectors that are closest to each input vector
encoding_idx = argmin(distances, dims=2)
encodings = zeros(Bool, size(distances, 1), num_embeddings)
encodings[encoding_idx] .= true;
# We can manually verify that the indices are correct. For example, check the distance
# between the first code vector to all embedding vectors:
ix_c = 1
argmin([norm(code[ix_c, :] - e.embedding[ix_e, :]) for ix_e ∈ 1:num_embeddings]) == encoding_idx[ix_c][2]

# Now quantize and reshape to size of original input size
quantized = reshape(encodings * e.embedding, (2, 2, batch_size, embedding_dim))


"""
    VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

Quantizes a tensor over a discrete embedding space.
Adapted from https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb

The output tensor will have the same dimension as the input
"""

struct VectorQuantizer
    embedding_dim:: Int
end