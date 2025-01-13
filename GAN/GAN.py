import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

"""#Exercise 1: Implementing a Basic GAN

##Define the Generator and Discriminator
"""

def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(28 * 28, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation='leaky_relu'),
        layers.Dense(256, activation='leaky_relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

"""##Training the GAN"""

def train_gan(latent_dim, epochs, batch_size):
    # Load and preprocess the MNIST dataset
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
    x_train = np.expand_dims(x_train, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)

    # Build models
    generator = build_generator(latent_dim)
    generator.summary()

    discriminator = build_discriminator()
    discriminator.summary()
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Training step
    @tf.function
    def train_step(real_images):
        batch_size = tf.shape(real_images)[0]

        # Generate noise and fake images
        noise = tf.random.normal([batch_size, latent_dim])
        fake_images = generator(noise, training=True)

        # Labels for real and fake images
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # Train the discriminator
        with tf.GradientTape() as tape:
            real_predictions = discriminator(real_images, training=True)
            fake_predictions = discriminator(fake_images, training=True)

            d_loss_real = loss_fn(real_labels, real_predictions)
            d_loss_fake = loss_fn(fake_labels, fake_predictions)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape:
            noise = tf.random.normal([batch_size, latent_dim])
            fake_images = generator(noise, training=True)
            fake_predictions = discriminator(fake_images, training=True)
            g_loss = loss_fn(real_labels, fake_predictions)

        g_gradients = tape.gradient(g_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        return d_loss, g_loss

    # Training loop
    for epoch in range(epochs):
      epoch_d_loss = 0
      epoch_g_loss = 0
      num_batches = 0

      for real_images in dataset:
          d_loss, g_loss = train_step(real_images)
          epoch_d_loss += d_loss
          epoch_g_loss += g_loss
          num_batches += 1

      # Average losses for the epoch
      epoch_d_loss /= num_batches
      epoch_g_loss /= num_batches
      d_losses.append(epoch_d_loss.numpy())
      g_losses.append(epoch_g_loss.numpy())

      print(f"Epoch {epoch + 1}/{epochs} | Average D Loss: {epoch_d_loss.numpy():.4f} | Average G Loss: {epoch_g_loss.numpy():.4f}")

      # Visualize generated images every 10 epochs
      if (epoch + 1) % 10 == 0:
          visualize_generated_images(generator, latent_dim, 10)

    plot_loss_curves(d_losses, g_losses)

"""##Visualizing Images"""

def visualize_generated_images(generator, latent_dim, num_classes):
    fig, axs = plt.subplots(1, num_classes, figsize=(10, 2))
    for i in range(num_classes):
        noise = np.random.normal(0, 1, (1, latent_dim))
        label = tf.keras.utils.to_categorical([i], num_classes)
        generated_image = generator.predict([noise, label])[0]
        generated_image = 0.5 * generated_image + 0.5  # Rescale to [0, 1]

        axs[i].imshow(generated_image[:, :, 0], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(str(i))
    plt.show()


def visualize_real_images(num_classes=10, num_rows=3):
    # Load the MNIST dataset
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

    # Normalize images to the range [0, 1] for visualization
    x_train = x_train.astype('float32') / 255.0

    # Initialize a figure
    fig, axs = plt.subplots(num_rows, num_classes, figsize=(10, 3))

    for row in range(num_rows):
        for i in range(num_classes):
            # Find the index of the first occurrence of the label (i) for this row
            idx = np.where(y_train == i)[0][row]
            real_image = x_train[idx]

            # Plot the real image
            axs[row, i].imshow(real_image, cmap='gray')
            axs[row, i].axis('off')
            if row == 0:  # Only set title for the first row
                axs[row, i].set_title(str(i))

    plt.tight_layout()
    plt.show()

"""##Plot loss curves"""

def plot_loss_curves(d_losses, g_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

"""##Run it"""

# Lists to store losses
d_losses = []
g_losses = []

latent_dim = 100
train_gan(latent_dim=latent_dim, epochs=50, batch_size=64)

visualize_real_images()

"""#Exercise 2: Implementing a Conditional GAN (cGAN)

##Define the Conditional Generator and the Conditional Discriminator
"""

def build_conditional_generator(latent_dim, num_classes):
    label_input = layers.Input(shape=(num_classes,))
    noise_input = layers.Input(shape=(latent_dim,))

    x = layers.Concatenate()([noise_input, label_input])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(28 * 28, activation='tanh')(x)
    img_output = layers.Reshape((28, 28, 1))(x)

    return tf.keras.Model([noise_input, label_input], img_output, name="Generator")

# Build the discriminator for cGAN
def build_conditional_discriminator(num_classes):
    label_input = layers.Input(shape=(num_classes,))
    img_input = layers.Input(shape=(28, 28, 1))

    x = layers.Flatten()(img_input)
    x = layers.Concatenate()([x, label_input])
    x = layers.Dense(512, activation=layers.LeakyReLU(alpha=0.2))(x)
    x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.2))(x)
    validity_output = layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model([img_input, label_input], validity_output, name="Discriminator")

"""##Training the Conditional GAN"""

def train_conditional_gan(latent_dim, num_classes, epochs, batch_size):
    # Load and preprocess the MNIST dataset
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
    x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

    # Create a dataset object
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size)

    # Build models
    generator = build_conditional_generator(latent_dim, num_classes)
    generator.summary()

    discriminator = build_conditional_discriminator(num_classes)
    discriminator.summary()
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Training step function with @tf.function
    @tf.function
    def train_step(real_imgs, real_labels):
        batch_size = tf.shape(real_imgs)[0]

        # Generate fake images
        noise = tf.random.normal([batch_size, latent_dim])
        fake_imgs = generator([noise, real_labels], training=True)

        # Labels for real and fake images
        real_validity = tf.ones((batch_size, 1))
        fake_validity = tf.zeros((batch_size, 1))

        # Train the discriminator
        with tf.GradientTape() as tape:
            real_predictions = discriminator([real_imgs, real_labels], training=True)
            fake_predictions = discriminator([fake_imgs, real_labels], training=True)

            d_loss_real = loss_fn(real_validity, real_predictions)
            d_loss_fake = loss_fn(fake_validity, fake_predictions)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape:
            noise = tf.random.normal([batch_size, latent_dim])
            fake_imgs = generator([noise, real_labels], training=True)
            fake_predictions = discriminator([fake_imgs, real_labels], training=True)
            g_loss = loss_fn(real_validity, fake_predictions)

        g_gradients = tape.gradient(g_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        return d_loss, g_loss

    # Training loop
    for epoch in range(epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        num_batches = 0

        for real_imgs, real_labels in dataset:
            d_loss, g_loss = train_step(real_imgs, real_labels)
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            num_batches += 1

        # Average losses for the epoch
        epoch_d_loss /= num_batches
        epoch_g_loss /= num_batches
        d_losses.append(epoch_d_loss.numpy())
        g_losses.append(epoch_g_loss.numpy())


        print(f"Epoch {epoch + 1}/{epochs} | Average D Loss: {epoch_d_loss.numpy():.4f} | Average G Loss: {epoch_g_loss.numpy():.4f}")

        # Visualize generated images every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualize_conditional_images(generator, latent_dim, num_classes)

    # Plot loss curves
    plot_loss_curves(d_losses, g_losses)

"""##Visulaize Images"""

def visualize_conditional_images(generator, latent_dim, num_classes):
    fig, axs = plt.subplots(1, num_classes, figsize=(10, 2))
    for i in range(num_classes):
        noise = np.random.normal(0, 1, (1, latent_dim))
        label = tf.keras.utils.to_categorical([i], num_classes)
        generated_image = generator.predict([noise, label])[0]
        generated_image = 0.5 * generated_image + 0.5  # Rescale to [0, 1]

        axs[i].imshow(generated_image[:, :, 0], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(str(i))
    plt.show()


def visualize_real_images(num_classes=10, num_rows=3):
    # Load the MNIST dataset
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

    # Normalize images to the range [0, 1] for visualization
    x_train = x_train.astype('float32') / 255.0

    # Initialize a figure
    fig, axs = plt.subplots(num_rows, num_classes, figsize=(10, 3))

    for row in range(num_rows):
        for i in range(num_classes):
            # Find the index of the first occurrence of the label (i) for this row
            idx = np.where(y_train == i)[0][row]
            real_image = x_train[idx]

            # Plot the real image
            axs[row, i].imshow(real_image, cmap='gray')
            axs[row, i].axis('off')
            if row == 0:  # Only set title for the first row
                axs[row, i].set_title(str(i))

    plt.tight_layout()
    plt.show()

"""##Plot loss curves"""

def plot_loss_curves(d_losses, g_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

"""##Run it"""

# Lists to store losses
d_losses = []
g_losses = []

latent_dim = 100
train_conditional_gan(latent_dim=latent_dim, num_classes=10, epochs=50, batch_size=64)

visualize_real_images()