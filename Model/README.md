# Guidance for Parameter Tuning

All parameters are organised in `Setting.py`, the settings within which can be found in the following document. You can use different setting files to experiment with different settings.

All setting names are capitalised in the setting file, and all spaces should be replaced by an underscore.

## Global

| Setting | Description |
| ------- | ----------- |
| project root | We assume the application always starts at the project root. |
| time window allocation increment | Specifies, when need to preallocate memory for time steps, grow the array by multiple of time window size. Set to a lager number reduces the number of allocation, but may increase memory consumption. |

## Special Token

| Setting | Description |
| ------- | ----------- |
| SOS | Start Of Sequence |
| EOS | End Of Sequence |
| PAD | Padding |

## Embedding

| Setting | Description |
| ------- | ----------- |
| note original feature size | The feature size before embedding note. 128 velocity/control value + all special tokens. |
| note embedding feature size | The feature size after embedding note. |
| time embedding layer kernel | The kernel size of each layer to embed time; the product must equal to the time window size. |
| time window size | The number of time step to be grouped together into one feature, for dimensionality reduction. |
| embedded feature size | The number of element in the feature vector after embedding an input feature, including both note and time. |
| max sequence length | The maximum number of time window (grouped time step) the model can take. |

## Transformer

| Setting | Description |
| ------- | ----------- |
| attention head count | The number of head used in multi-head attention model, by splitting the embedded feature. |
| feed forward latent size | The hidden layer dimension of the feed forward MLP. |
| coder layer count | The number of hidden layers for encoder and decoder. |
| causal attention mask | Indicate whether to apply causal mask instead of the explicitly generated attention. |

## Discriminator

| Setting | Description |
| ------- | ----------- |
| time kernel size | The kernel size of each layer. Must be at least 2 sizes specified. Stride is half of the kernel, padding is half of the stride; except the last layer, which will have full stride and no padding. The input will be an integer multiple of the time window size, and output should be one. Consult PyTorch documentation to calculate the size of output of each layer. |
| time layer feature | The number of convolutional layer to extract time information. Must have one less member than the number of element. The input and output layers always have feature size of one. |
| sequence hidden | The hidden layer dimension for the sequence extraction network. |
| sequence layer | The number of layer of the sequence network. |
| leaky slope | The slope factor used by leaky ReLU in each convolution layer. |

## Dataset

Please change the path variables so that it matches your environment. As a convention to avoid confusion, please alway use absolute path.

| Setting | Description |
| ------- | ----------- |
| data shuffle seed | Seed used for randomly permuting the dataset. |
| data split | Proportion of [train, validation, test]; must sum up to 1.0. |
| ASAP path | Path to the root of [ASAP](https://github.com/fosfrancesco/asap-dataset) dataset. |
| midi cache path | Intermediate MIDI file cache output directory. |
| model output path | Hold binary of the trained model. |
| train stats log path | Store training stats such as training accuracy and loss. |

## Training

| Setting | Description |
| ------- | ----------- |
| epoch | The number of epoch. |
| batch size | The size of batch. |
| lr generator | The learning rate for the generator. |
| lr discriminator | The learning rate for the discriminator. |
| beta generator | The beta parameter for the generator. |
| beta discriminator | The beta parameter for the discriminator. |
| log frequency | Specify logging frequency in term of number of iteration elapsed. |

## Dropout

Specifies the dropout probability for each component. Their names are pretty self-explanatory.

- position embedding
- full embedding
- coder
- discriminator sequence