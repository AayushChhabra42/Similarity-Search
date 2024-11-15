a = "purple is the best city in the forest"
b = "there is an art to getting your way and throwing bananas on to the street is not it"  # this is very similar to 'g'
c = "it is not often you find soggy bananas on the street"
d = "green should have smelled more tranquil but somehow it just tasted rotten"
e = "joyce enjoyed eating pancakes with ketchup"
f = "as the asteroid hurtled toward earth becky was upset her dentist appointment had been canceled"
g = "to get your way you must not bombard the road with yellow fruit"  # this is very similar to 'b'

from transformers import AutoTokenizer,AutoModel
import torch

tokenizer=AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

tokens = tokenizer([a, b, c, d, e, f, g],max_length=128,truncation=True,padding='max_length',return_tensors='pt')

print(tokens.keys())

print(tokens['input_ids'][0])

output=model(**tokens)
print(output.keys())

embeddings = output.last_hidden_state
print(embeddings[0])
print(embeddings[0].shape)

#Since we have a vector representation for each token in our sentence, we need to perform mean pooling to create a sentence vector from token vectors
#To do this multiply each value in our embeddings tensor by its respective attention_mask value. The attention_mask contains ones where we have ‘real tokens’ (eg not padding tokens), and zeros elsewhere — this operation allows us to ignore non-real tokens.

mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
print(mask.shape)
print(mask[0])

#Now we have a masking array that has an equal shape to our output `embeddings` - we multiply those together to apply the masking operation on our outputs.

masked_embeddings = embeddings * mask
print(masked_embeddings[0])

#Sum the remaining embeddings along axis 1 to get a total value in each of our 768 values.

summed = torch.sum(masked_embeddings, 1)
print(summed.shape)

#Next, we count the number of values that should be given attention in each position of the tensor (+1 for real tokens, +0 for non-real).

counted = torch.clamp(mask.sum(1), min=1e-9)
print(counted.shape)

#Finally, we get our mean-pooled values as the `summed` embeddings divided by the number of values that should be given attention, `counted`.

mean_pooled = summed / counted
print(mean_pooled.shape)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# convert to numpy array from torch tensor
mean_pooled = mean_pooled.detach().numpy()

# calculate similarities (will store in array)
scores = np.zeros((mean_pooled.shape[0], mean_pooled.shape[0]))
for i in range(mean_pooled.shape[0]):
    scores[i, :] = cosine_similarity(
        [mean_pooled[i]],
        mean_pooled
    )[0]

print(scores)