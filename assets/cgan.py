"""
Conditional Generative Adversarial Network (CGAN) implementation for MNIST digit generation.
Uses Keras framework with TensorFlow backend.
"""

from __future__ import print_function, division

import os
import sys
import traceback
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Embedding
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

class CGAN():
    """CGAN class implementation"""
    
    def __init__(self):
        """Initialize CGAN with architecture parameters and build models"""
        
        # Image configuration
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10  # MNIST digits (0-9)
        self.latent_dim = 100  # Noise vector size

        # Configure optimizer
        optimizer = Adam(0.0002, 0.5)

        try:
            # Build and compile discriminator
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(
                loss=['binary_crossentropy'],
                optimizer=optimizer,
                metrics=['accuracy']
            )

            # Build generator
            self.generator = self.build_generator()

            # Combined model (generator -> discriminator)
            noise = Input(shape=(self.latent_dim,))
            label = Input(shape=(1,))
            img = self.generator([noise, label])

            # Freeze discriminator during generator training
            self.discriminator.trainable = False
            valid = self.discriminator([img, label])

            # Compile combined model
            self.combined = Model([noise, label], valid)
            self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            traceback.print_exc()
            sys.exit(1)

    def build_generator(self):
        """Build generator model that maps (noise, label) -> image"""
        
        model = Sequential()
        # Foundation for 7x7 image
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        # Define model inputs
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        
        # Embed label and multiply with noise
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])
        
        # Generate image from combined input
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):
        """Build discriminator model that classifies (image, label) as real/fake"""
        
        model = Sequential()
        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        # Define model inputs
        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        # Process inputs
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)
        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        """Train CGAN model
        
        Args:
            epochs (int): Number of training iterations
            batch_size (int): Size of training batches
            sample_interval (int): Interval for saving generated samples
        """
        
        try:
            # Load and preprocess MNIST data
            (X_train, y_train), (_, _) = mnist.load_data()
            X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
            X_train = np.expand_dims(X_train, axis=3)
            y_train = y_train.reshape(-1, 1)

            # Adversarial labels
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            # Create output directory
            os.makedirs("images3", exist_ok=True)

            for epoch in range(epochs):
                try:
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    
                    # Select random batch
                    idx = np.random.randint(0, X_train.shape[0], batch_size)
                    imgs, labels = X_train[idx], y_train[idx]

                    # Generate fake images
                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                    gen_imgs = self.generator.predict([noise, labels])

                    # Train discriminator
                    d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
                    d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    # ---------------------
                    #  Train Generator
                    # ---------------------
                    
                    # Generate random labels
                    sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
                    
                    # Train generator (to fool discriminator)
                    g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

                    # Progress report
                    print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

                    # Save samples
                    if epoch % sample_interval == 0:
                        self.sample_images(epoch)
                        
                except KeyboardInterrupt:
                    print("\nTraining interrupted by user")
                    sys.exit(0)
                except Exception as e:
                    print(f"Error during epoch {epoch}: {str(e)}")
                    traceback.print_exc()

        except Exception as e:
            print(f"Training failed: {str(e)}")
            traceback.print_exc()
            sys.exit(1)

    def sample_images(self, epoch):
        """Save generated images with labels
        
        Args:
            epoch (int): Current epoch number for filename
        """
        
        # Generate grid of images
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        try:
            gen_imgs = self.generator.predict([noise, sampled_labels])
            gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0,1]

            # Plot configuration
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                    axs[i,j].set_title(f"Digit: {sampled_labels[cnt][0]}")
                    axs[i,j].axis('off')
                    cnt += 1
                    
            # Save figure
            fig.savefig(f"images3/{epoch}.png")
            plt.close()
            
        except Exception as e:
            print(f"Error saving images: {str(e)}")
            traceback.print_exc()


if __name__ == '__main__':
    try:
        cgan = CGAN()
        cgan.train(epochs=20000, batch_size=32, sample_interval=200)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
        