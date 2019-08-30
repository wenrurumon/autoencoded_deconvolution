#setup
reticulate::use_python('/usr/bin/python3')
library('keras')
library('tensorflow')
K <- backend()

# Parameters --------------------------------------------------------------

batch_size <- 256L
# this is the size of our encoded representations
original_dim <- 784L
latent_dim <- 784L
encoding_dim <- 32L
#epochs <- 50L 

# Model definition --------------------------------------------------------
# this is our input placeholder
input_img <- layer_input(shape = c(original_dim))
# "encoded" is the encoded representation of the input
encoded<- layer_dense(input_img,encoding_dim , activation = "relu")
# "decoded" is the lossy reconstruction of the input
decoded<- layer_dense(encoded, latent_dim , activation = "sigmoid")

# this model maps an input to its reconstruction
autoencoder <- keras_model(input_img, decoded)
#Let's also create a separate encoder model:
# this model maps an input to its encoded representation
encoder <- keras_model(input_img, encoded)

#As well as the decoder model:
# create a placeholder for an encoded (32-dimensional) input
encoded_input <- layer_input(shape = c(encoding_dim))
# retrieve the last layer of the autoencoder model
decoder_layer = get_layer(autoencoder,index=-1)
# create the decoder model
decoder <- keras_model(encoded_input, decoder_layer(encoded_input))

autoencoder  %>% compile(optimizer = 'adadelta', loss = 'binary_crossentropy')


mnist <- dataset_mnist()
x_train <- mnist$train$x/255
x_test <- mnist$test$x/255
x_train <- x_train %>% apply(1, as.numeric) %>% t()
x_test <- x_test %>% apply(1, as.numeric) %>% t()

autoencoder %>% fit (x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=TRUE,
                validation_data=list(x_test, x_test))
