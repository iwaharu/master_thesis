from layers import *
import tensorflow as tf
from node_embed_model import *

"""
Disclaimer: layers.py is included a modified (TF1->TF2) version
of the one defined in the following repository.
https://github.com/deezer/gravity_graph_autoencoders
"""

class GenerativeGCNAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(GenerativeGCNAE, self).__init__(**kwargs)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.adj_loss_tracker = tf.keras.metrics.Mean(name="adj_loss")
        self.feature_loss_tracker = tf.keras.metrics.Mean(name="feature_loss")
        self.accuracy_tracker = tf.keras.metrics.Mean(name="accuracy")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.adj_loss_tracker,
            self.feature_loss_tracker,
            self.accuracy_tracker
        ]

    def build(self, hyperparams):
        self.feature_input = tf.keras.Input(shape=hyperparams['feature_shape'], batch_size=hyperparams['batchsize'], name="feature_input")
        self.adj_input = tf.keras.Input(shape=hyperparams['adj_shape'], batch_size=hyperparams['batchsize'], name="adj_input")
        self.conv_1 = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                num_features = hyperparams['feature_shape'][1],
                                node_encode_dim = hyperparams['node_hidden_dim'],
                                dropout = hyperparams['dropout'],
                                act = tf.nn.relu,
                                name='conv_1')([self.feature_input,self.adj_input])

        self.z_mean = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                    num_features = hyperparams['node_hidden_dim'],
                                    node_encode_dim = hyperparams['node_embed_shape'][-1],
                                    dropout = hyperparams['dropout'],
                                    act = tf.keras.activations.linear,
                                    name='z_mean')([self.conv_1,self.adj_input])
        self.encoder = tf.keras.Model(inputs=[self.feature_input,self.adj_input], outputs=self.z_mean, name='encoder')
        #print(self.encoder.summary())

        self.decoder_input = tf.keras.Input(shape=hyperparams['node_embed_shape'], name='node_embedding')
        self.output_adj = InnerProductDecoder(dropout=hyperparams['dropout'], act=tf.keras.activations.linear, name="output_adj")(self.decoder_input)
        self.output_feature = DenseFeatureDecoder(hyperparams['node_embed_shape'][-1], hyperparams['feature_shape'][1], \
                                    dropout=hyperparams['dropout'], act=tf.keras.activations.relu)(self.decoder_input)
        self.decoder = tf.keras.Model(inputs=self.decoder_input, outputs=[self.output_adj, self.output_feature], name="decoder")
        #print(self.decoder.summary())

        self.batch_size = hyperparams['batchsize']
        self.built = True

    def call(self, inputs, *args, **kwargs):
        x, _ = inputs
        z_mean = self.encoder(x)
        adj, feature = self.decoder(z_mean)
        return adj, feature

    def train_step(self, inputs):
        x, y = inputs
        fmx, _ = x
        target, norm, pos_weight = y

        with tf.GradientTape() as tape:
            z_mean = self.encoder(x)
            adj, feature = self.decoder(z_mean)

            adj = tf.reshape(adj, [self.batch_size, -1])
            target = tf.reshape(target, [self.batch_size, -1])
            pos_weight = pos_weight[:,tf.newaxis]

            adj_loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits( \
                                    logits=adj, labels=target, pos_weight=pos_weight), axis=1)
            feature_loss = norm * tf.reduce_mean(tf.keras.metrics.binary_crossentropy(fmx, feature), axis=1)
            total_loss = adj_loss + feature_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        correct_pred = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(adj), 0.5), tf.int32),
                        tf.cast(target, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.total_loss_tracker.update_state(total_loss)
        self.adj_loss_tracker.update_state(adj_loss)
        self.feature_loss_tracker.update_state(feature_loss)
        self.accuracy_tracker.update_state(accuracy)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "adj_loss": self.adj_loss_tracker.result(),
            "feature_loss": self.feature_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result()
        }

    def test_step(self, inputs):
        x, y = inputs
        fmx, _ = x
        target, norm, pos_weight = y

        z_mean = self.encoder(x)
        adj, feature = self.decoder(z_mean)

        adj = tf.reshape(adj, [self.batch_size, -1])
        target = tf.reshape(target, [self.batch_size, -1])
        pos_weight = pos_weight[:,tf.newaxis]

        adj_loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits( \
                                    logits=adj, labels=target, pos_weight=pos_weight), axis=1)
        feature_loss = norm * tf.reduce_mean(tf.keras.metrics.binary_crossentropy(fmx, feature), axis=1)
        total_loss = adj_loss + feature_loss

        correct_pred = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(adj), 0.5), tf.int32),
                        tf.cast(target, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.total_loss_tracker.update_state(total_loss)
        self.adj_loss_tracker.update_state(adj_loss)
        self.feature_loss_tracker.update_state(feature_loss)
        self.accuracy_tracker.update_state(accuracy)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "adj_loss": self.adj_loss_tracker.result(),
            "feature_loss": self.feature_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result()
        }

    def encode(self, data):
        x, _ = data
        z_mean = self.encoder.predict(x)
        return z_mean

class GraphEmbedGCNAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(GraphEmbedGCNAE, self).__init__(**kwargs)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.adj_loss_tracker = tf.keras.metrics.Mean(name="adj_loss")
        self.feature_loss_tracker = tf.keras.metrics.Mean(name="feature_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.adj_loss_tracker,
            self.feature_loss_tracker,
        ]

    def build(self, hyperparams):
        feature_input = tf.keras.Input(shape=hyperparams['feature_shape'], batch_size=hyperparams['batchsize'], name="feature_input")
        adj_input = tf.keras.Input(shape=hyperparams['adj_shape'], batch_size=hyperparams['batchsize'], name="adj_input")
        conv_1 = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                num_features = hyperparams['feature_shape'][1],
                                node_encode_dim = hyperparams['node_hidden_dim'],
                                dropout = hyperparams['dropout'],
                                act = tf.nn.relu,
                                name='conv_1')([feature_input,adj_input])

        z_mean = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                    num_features = hyperparams['node_hidden_dim'],
                                    node_encode_dim = hyperparams['node_embed_shape'][-1],
                                    dropout = hyperparams['dropout'],
                                    act = tf.keras.activations.linear,
                                    name='z_mean')([conv_1,adj_input])
        self.node_encoder = tf.keras.Model(inputs=[feature_input,adj_input], outputs=z_mean, name='node_encoder')
        #print(self.node_encoder.summary())

        node_embed_input = tf.keras.Input(shape=hyperparams['node_embed_shape'], batch_size=hyperparams['batchsize'], name="node_embed_input")
        conv_2 = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='relu')(node_embed_input)
        flatten_z_mean = tf.keras.layers.Flatten()(conv_2)
        encode_dense_1 = tf.keras.layers.Dense(hyperparams['graph_hidden_dim'], activation='relu')(flatten_z_mean)
        encode_dense_2 = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], activation='relu')(encode_dense_1)
        self.graph_encoder = tf.keras.Model(inputs=node_embed_input, outputs=encode_dense_2, name='graph_encoder')
        #print(self.graph_encoder.summary())

        graph_decode_input = tf.keras.Input(shape=hyperparams['graph_embed_dim'], name='graph_embedding')
        decode_dense_1 = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], activation='relu')(graph_decode_input)
        repeat = tf.keras.layers.RepeatVector(hyperparams['feature_shape'][0])(decode_dense_1)
        gru_1 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(repeat)
        gru_2 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_1)
        gru_3 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_2)
        decode_dense_2 = tf.keras.layers.Dense(hyperparams['node_embed_shape'][-1], activation='relu')(gru_3)
        self.graph_decoder = tf.keras.Model(inputs=graph_decode_input, outputs=decode_dense_2, name='graph_decoder')
        #print(self.graph_decoder.summary())
        
        node_decode_input = tf.keras.Input(shape=hyperparams['node_embed_shape'], name='node_embedding')
        output_adj = InnerProductDecoder(dropout=hyperparams['dropout'], act=tf.keras.activations.linear, name="output_adj")(node_decode_input)
        output_feature = DenseFeatureDecoder(hyperparams['node_embed_shape'][-1], hyperparams['feature_shape'][1], \
                                    dropout=hyperparams['dropout'], act=tf.keras.activations.relu)(node_decode_input)
        self.node_decoder = tf.keras.Model(inputs=node_decode_input, outputs=[output_adj, output_feature], name="node_decoder")
        #print(self.node_decoder.summary())

        self._set_inputs(inputs=[feature_input,adj_input], outputs=[output_adj, output_feature])

        self.batch_size = hyperparams['batchsize']
        self.built = True

    def call(self, inputs, *args, **kwargs):
        x, _ = inputs
        node_z_mean = self.node_encoder(x)
        graph_z_mean = self.graph_encoder(node_z_mean)

        graph_reconstructed = self.graph_decoder(graph_z_mean)
        adj, feature = self.node_decoder(graph_reconstructed)
        return adj, feature

    def encode(self, inputs):
        x, _ = inputs
        node_z_mean = self.node_encoder(x)
        graph_z_mean = self.graph_encoder(node_z_mean)
        return graph_z_mean

    def decode(self, z):
        graph_reconstructed = self.graph_decoder(z)
        adj, feature = self.node_decoder(graph_reconstructed)
        return adj, feature

    @tf.function
    def train_step(self, inputs):
        x, y = inputs
        fmx, _ = x
        target, norm, pos_weight = y

        with tf.GradientTape() as tape:
            node_z_mean = self.node_encoder(x)
            graph_z_mean = self.graph_encoder(node_z_mean)

            graph_reconstructed = self.graph_decoder(graph_z_mean)
            adj, feature = self.node_decoder(graph_reconstructed)

            adj = tf.reshape(adj, [self.batch_size, -1])
            target = tf.reshape(target, [self.batch_size, -1])
            pos_weight = pos_weight[:,tf.newaxis]

            adj_loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits( \
                                    logits=adj, labels=target, pos_weight=pos_weight), axis=1)
            feature_loss = norm * tf.reduce_mean(tf.keras.metrics.binary_crossentropy(fmx, feature), axis=1)
            total_loss = adj_loss + feature_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.adj_loss_tracker.update_state(adj_loss)
        self.feature_loss_tracker.update_state(feature_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "adj_loss": self.adj_loss_tracker.result(),
            "feature_loss": self.feature_loss_tracker.result()
        }

    @tf.function
    def test_step(self, inputs):
        x, y = inputs
        fmx, _ = x
        target, norm, pos_weight = y

        node_z_mean = self.node_encoder(x)
        graph_z_mean = self.graph_encoder(node_z_mean)

        graph_reconstructed = self.graph_decoder(graph_z_mean)
        adj, feature = self.node_decoder(graph_reconstructed)

        adj = tf.reshape(adj, [self.batch_size, -1])
        target = tf.reshape(target, [self.batch_size, -1])
        pos_weight = pos_weight[:,tf.newaxis]

        adj_loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits( \
                                    logits=adj, labels=target, pos_weight=pos_weight), axis=1)
        feature_loss = norm * tf.reduce_mean(tf.keras.metrics.binary_crossentropy(fmx, feature), axis=1)
        total_loss = adj_loss + feature_loss

        self.total_loss_tracker.update_state(total_loss)
        self.adj_loss_tracker.update_state(adj_loss)
        self.feature_loss_tracker.update_state(feature_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "adj_loss": self.adj_loss_tracker.result(),
            "feature_loss": self.feature_loss_tracker.result()
        }

class GraphEmbedGCNVAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(GraphEmbedGCNVAE, self).__init__(**kwargs)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.adj_loss_tracker = tf.keras.metrics.Mean(name="adj_loss")
        self.feature_loss_tracker = tf.keras.metrics.Mean(name="feature_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.adj_loss_tracker,
            self.feature_loss_tracker,
            self.kl_loss_tracker
        ]

    def build(self, hyperparams):
        feature_input = tf.keras.Input(shape=hyperparams['feature_shape'], batch_size=hyperparams['batchsize'], name="feature_input")
        adj_input = tf.keras.Input(shape=hyperparams['adj_shape'], batch_size=hyperparams['batchsize'], name="adj_input")
        conv_1 = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                num_features = hyperparams['feature_shape'][1],
                                node_encode_dim = hyperparams['node_hidden_dim'],
                                dropout = hyperparams['dropout'],
                                act = tf.nn.relu,
                                name='conv_1')([feature_input,adj_input])

        node_z_mean = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                    num_features = hyperparams['node_hidden_dim'],
                                    node_encode_dim = hyperparams['node_embed_shape'][-1],
                                    dropout = hyperparams['dropout'],
                                    act = tf.keras.activations.linear,
                                    name='node_z_mean')([conv_1,adj_input])
        self.node_encoder = tf.keras.Model(inputs=[feature_input,adj_input], outputs=node_z_mean, name='node_encoder')
        #print(self.node_encoder.summary())

        node_embed_input = tf.keras.Input(shape=hyperparams['node_embed_shape'], batch_size=hyperparams['batchsize'], name="node_embed_input")
        conv_2 = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='relu')(node_embed_input)
        flatten_z_mean = tf.keras.layers.Flatten()(conv_2)
        encode_dense_1 = tf.keras.layers.Dense(hyperparams['graph_hidden_dim'], activation='relu')(flatten_z_mean)

        graph_z_mean = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], activation='relu', name="graph_z_mean")(encode_dense_1)
        graph_z_log_var = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], name="graph_z_log_var")(encode_dense_1)
        graph_z = Sampling(name="graph_z")([graph_z_mean, graph_z_log_var])

        self.graph_encoder = tf.keras.Model(inputs=node_embed_input, outputs=[graph_z_mean, graph_z_log_var, graph_z], name='graph_encoder')
        #print(self.graph_encoder.summary())

        graph_decode_input = tf.keras.Input(shape=hyperparams['graph_embed_dim'], name='graph_embedding')
        decode_dense_1 = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], activation='relu')(graph_decode_input)
        repeat = tf.keras.layers.RepeatVector(hyperparams['feature_shape'][0])(decode_dense_1)
        gru_1 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(repeat)
        gru_2 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_1)
        gru_3 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_2)
        decode_dense_2 = tf.keras.layers.Dense(hyperparams['node_embed_shape'][-1], activation='relu')(gru_3)
        self.graph_decoder = tf.keras.Model(inputs=graph_decode_input, outputs=decode_dense_2, name='graph_decoder')
        #print(self.graph_decoder.summary())
        
        node_decode_input = tf.keras.Input(shape=hyperparams['node_embed_shape'], name='node_embedding')
        output_adj = InnerProductDecoder(dropout=hyperparams['dropout'], act=tf.keras.activations.linear, name="output_adj")(node_decode_input)
        output_feature = DenseFeatureDecoder(hyperparams['node_embed_shape'][-1], hyperparams['feature_shape'][1], \
                                    dropout=hyperparams['dropout'], act=tf.keras.activations.relu)(node_decode_input)
        self.node_decoder = tf.keras.Model(inputs=node_decode_input, outputs=[output_adj, output_feature], name="node_decoder")
        #print(self.node_decoder.summary())

        self._set_inputs(inputs=[feature_input,adj_input], outputs=[output_adj, output_feature])

        self.num_node = hyperparams['feature_shape'][0]
        self.batch_size = hyperparams['batchsize']
        self.weight_ratio = hyperparams['weight_ratio']
        self.built = True

    def call(self, inputs, *args, **kwargs):
        x, _ = inputs

        node_z_mean = self.node_encoder(x)
        z_mean, z_log_std, z = self.graph_encoder(node_z_mean)

        graph_reconstructed = self.graph_decoder(z)
        adj, feature = self.node_decoder(graph_reconstructed)
        return adj, feature

    def encode(self, inputs):
        x, _ = inputs
        node_z_mean = self.node_encoder.predict(x, batch_size=self.batch_size)
        z_mean, z_log_std, z = self.graph_encoder.predict(node_z_mean, batch_size=self.batch_size)
        return z

    def decode(self, z):
        graph_reconstructed = self.graph_decoder.predict(z, batch_size=self.batch_size)
        adj, feature = self.node_decoder.predict(graph_reconstructed, batch_size=self.batch_size)
        return adj, feature

    @tf.function
    def train_step(self, inputs):
        x, y = inputs
        fmx, _ = x
        target, norm, pos_weight = y

        with tf.GradientTape() as tape:
            node_z_mean = self.node_encoder(x)
            z_mean, z_log_std, z = self.graph_encoder(node_z_mean)

            graph_reconstructed = self.graph_decoder(z)
            adj, feature = self.node_decoder(graph_reconstructed)

            adj = tf.reshape(adj, [self.batch_size, -1])
            target = tf.reshape(target, [self.batch_size, -1])
            pos_weight = pos_weight[:,tf.newaxis]

            adj_loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits( \
                                    logits=adj, labels=target, pos_weight=pos_weight), axis=1)
            feature_loss = norm * tf.reduce_mean(tf.keras.metrics.binary_crossentropy(fmx, feature), axis=1)
            kl_loss = (0.5 / self.num_node) * tf.reduce_sum((1 + z_log_std - tf.square(z_mean) - tf.exp(z_log_std)), axis=1)
            reconstruction_loss = adj_loss + feature_loss
            total_loss = tf.reduce_mean(self.weight_ratio*reconstruction_loss - (1-self.weight_ratio)*kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.adj_loss_tracker.update_state(adj_loss)
        self.feature_loss_tracker.update_state(feature_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "adj_loss": self.adj_loss_tracker.result(),
            "feature_loss": self.feature_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    @tf.function
    def test_step(self, inputs):
        x, y = inputs
        fmx, _ = x
        target, norm, pos_weight = y

        node_z_mean = self.node_encoder(x)
        z_mean, z_log_std, z = self.graph_encoder(node_z_mean)

        graph_reconstructed = self.graph_decoder(z)
        adj, feature = self.node_decoder(graph_reconstructed)

        adj = tf.reshape(adj, [self.batch_size, -1])
        target = tf.reshape(target, [self.batch_size, -1])
        pos_weight = pos_weight[:,tf.newaxis]

        adj_loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits( \
                                    logits=adj, labels=target, pos_weight=pos_weight), axis=1)
        feature_loss = norm * tf.reduce_mean(tf.keras.metrics.binary_crossentropy(fmx, feature), axis=1)
        kl_loss = (0.5 / self.num_node) * tf.reduce_sum((1 + z_log_std - tf.square(z_mean) - tf.exp(z_log_std)), axis=1)
        reconstruction_loss = adj_loss + feature_loss
        total_loss = tf.reduce_mean(self.weight_ratio*reconstruction_loss - (1-self.weight_ratio)*kl_loss)

        self.total_loss_tracker.update_state(total_loss)
        self.adj_loss_tracker.update_state(adj_loss)
        self.feature_loss_tracker.update_state(feature_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "adj_loss": self.adj_loss_tracker.result(),
            "feature_loss": self.feature_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

class GraphEmbedSourceTargetGCNAE(GraphEmbedGCNAE):
    def __init__(self, **kwargs):
        super(GraphEmbedSourceTargetGCNAE, self).__init__(**kwargs)

    def build(self, hyperparams):
        feature_input = tf.keras.Input(shape=hyperparams['feature_shape'], batch_size=hyperparams['batchsize'], name="feature_input")
        adj_input = tf.keras.Input(shape=hyperparams['adj_shape'], batch_size=hyperparams['batchsize'], name="adj_input")
        conv_1 = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                num_features = hyperparams['feature_shape'][1],
                                node_encode_dim = hyperparams['node_hidden_dim'],
                                dropout = hyperparams['dropout'],
                                act = tf.nn.relu,
                                name='conv_1')([feature_input,adj_input])

        z_mean = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                    num_features = hyperparams['node_hidden_dim'],
                                    node_encode_dim = hyperparams['node_embed_shape'][-1],
                                    dropout = hyperparams['dropout'],
                                    act = tf.keras.activations.linear,
                                    name='z_mean')([conv_1,adj_input])
        self.node_encoder = tf.keras.Model(inputs=[feature_input,adj_input], outputs=z_mean, name='node_encoder')
        #print(self.node_encoder.summary())

        node_embed_input = tf.keras.Input(shape=hyperparams['node_embed_shape'], batch_size=hyperparams['batchsize'], name="node_embed_input")
        x = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='relu')(node_embed_input)
        x = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='relu')(x)
        x = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        encode_dense_1 = tf.keras.layers.Dense(hyperparams['graph_hidden_dim'], activation='relu')(x)
        encode_dense_2 = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], activation='relu')(encode_dense_1)
        self.graph_encoder = tf.keras.Model(inputs=node_embed_input, outputs=encode_dense_2, name='graph_encoder')
        #print(self.graph_encoder.summary())

        graph_decode_input = tf.keras.Input(shape=hyperparams['graph_embed_dim'], name='graph_embedding')
        decode_dense_1 = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], activation='relu')(graph_decode_input)
        repeat = tf.keras.layers.RepeatVector(hyperparams['feature_shape'][0])(decode_dense_1)
        gru_1 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(repeat)
        gru_2 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_1)
        gru_3 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_2)
        decode_dense_2 = tf.keras.layers.Dense(hyperparams['node_embed_shape'][-1], activation='relu')(gru_3)
        self.graph_decoder = tf.keras.Model(inputs=graph_decode_input, outputs=decode_dense_2, name='graph_decoder')
        #print(self.graph_decoder.summary())
        
        node_decode_input = tf.keras.Input(shape=hyperparams['node_embed_shape'], name='node_embedding')
        output_adj = SourceTargetInnerProductDecoder(\
                                node_encode_dim = hyperparams['node_embed_shape'][-1], \
                                dropout=hyperparams['dropout'], act=tf.keras.activations.linear)(node_decode_input)
        output_feature = DenseFeatureDecoder(hyperparams['node_embed_shape'][-1], hyperparams['feature_shape'][1], \
                                    dropout=hyperparams['dropout'], act=tf.keras.activations.relu)(node_decode_input)
        self.node_decoder = tf.keras.Model(inputs=node_decode_input, outputs=[output_adj, output_feature], name="node_decoder")
        #print(self.node_decoder.summary())

        self._set_inputs(inputs=[feature_input,adj_input], outputs=[output_adj, output_feature])

        self.batch_size = hyperparams['batchsize']
        self.built = True

class GraphEmbedSourceTargetGCNVAE(GraphEmbedGCNVAE):
    def __init__(self, **kwargs):
        super(GraphEmbedSourceTargetGCNVAE, self).__init__(**kwargs)

    def build(self, hyperparams):
        feature_input = tf.keras.Input(shape=hyperparams['feature_shape'], batch_size=hyperparams['batchsize'], name="feature_input")
        adj_input = tf.keras.Input(shape=hyperparams['adj_shape'], batch_size=hyperparams['batchsize'], name="adj_input")
        conv_1 = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                num_features = hyperparams['feature_shape'][1],
                                node_encode_dim = hyperparams['node_hidden_dim'],
                                dropout = hyperparams['dropout'],
                                act = tf.nn.relu,
                                name='conv_1')([feature_input,adj_input])

        node_z_mean = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                    num_features = hyperparams['node_hidden_dim'],
                                    node_encode_dim = hyperparams['node_embed_shape'][-1],
                                    dropout = hyperparams['dropout'],
                                    act = tf.keras.activations.linear,
                                    name='node_z_mean')([conv_1,adj_input])
        self.node_encoder = tf.keras.Model(inputs=[feature_input,adj_input], outputs=node_z_mean, name='node_encoder')
        #print(self.node_encoder.summary())

        node_embed_input = tf.keras.Input(shape=hyperparams['node_embed_shape'], batch_size=hyperparams['batchsize'], name="node_embed_input")
        x = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='relu')(node_embed_input)
        x = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='relu')(x)
        x = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        encode_dense_1 = tf.keras.layers.Dense(hyperparams['graph_hidden_dim'], activation='relu')(x)

        graph_z_mean = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], activation='relu', name="graph_z_mean")(encode_dense_1)
        graph_z_log_var = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], name="graph_z_log_var")(encode_dense_1)
        graph_z = Sampling(name="graph_z")([graph_z_mean, graph_z_log_var])

        self.graph_encoder = tf.keras.Model(inputs=node_embed_input, outputs=[graph_z_mean, graph_z_log_var, graph_z], name='graph_encoder')
        #print(self.graph_encoder.summary())

        graph_decode_input = tf.keras.Input(shape=hyperparams['graph_embed_dim'], name='graph_embedding')
        decode_dense_1 = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], activation='relu')(graph_decode_input)
        repeat = tf.keras.layers.RepeatVector(hyperparams['feature_shape'][0])(decode_dense_1)
        gru_1 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(repeat)
        gru_2 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_1)
        gru_3 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_2)
        decode_dense_2 = tf.keras.layers.Dense(hyperparams['node_embed_shape'][-1], activation='relu')(gru_3)
        self.graph_decoder = tf.keras.Model(inputs=graph_decode_input, outputs=decode_dense_2, name='graph_decoder')
        #print(self.graph_decoder.summary())
        
        node_decode_input = tf.keras.Input(shape=hyperparams['node_embed_shape'], name='node_embedding')
        output_adj = SourceTargetInnerProductDecoder(\
                                node_encode_dim = hyperparams['node_embed_shape'][-1], \
                                dropout=hyperparams['dropout'], act=tf.keras.activations.linear)(node_decode_input)
        output_feature = DenseFeatureDecoder(hyperparams['node_embed_shape'][-1], hyperparams['feature_shape'][1], \
                                    dropout=hyperparams['dropout'], act=tf.keras.activations.relu)(node_decode_input)
        self.node_decoder = tf.keras.Model(inputs=node_decode_input, outputs=[output_adj, output_feature], name="node_decoder")
        #print(self.node_decoder.summary())

        self._set_inputs(inputs=[feature_input,adj_input], outputs=[output_adj, output_feature])

        self.num_node = hyperparams['feature_shape'][0]
        self.batch_size = hyperparams['batchsize']
        self.weight_ratio = hyperparams['weight_ratio']
        self.built = True

class GraphEmbedGravityGCNAE(GraphEmbedGCNAE):
    def __init__(self, **kwargs):
        super(GraphEmbedGravityGCNAE, self).__init__(**kwargs)

    def build(self, hyperparams):
        feature_input = tf.keras.Input(shape=hyperparams['feature_shape'], batch_size=hyperparams['batchsize'], name="feature_input")
        adj_input = tf.keras.Input(shape=hyperparams['adj_shape'], batch_size=hyperparams['batchsize'], name="adj_input")
        conv_1 = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                num_features = hyperparams['feature_shape'][1],
                                node_encode_dim = hyperparams['node_hidden_dim'],
                                dropout = hyperparams['dropout'],
                                act = tf.nn.relu,
                                name='conv_1')([feature_input,adj_input])

        z_mean = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                    num_features = hyperparams['node_hidden_dim'],
                                    node_encode_dim = hyperparams['node_embed_shape'][-1],
                                    dropout = hyperparams['dropout'],
                                    act = tf.keras.activations.linear,
                                    name='z_mean')([conv_1,adj_input])
        self.node_encoder = tf.keras.Model(inputs=[feature_input,adj_input], outputs=z_mean, name='node_encoder')
        #print(self.node_encoder.summary())

        node_embed_input = tf.keras.Input(shape=hyperparams['node_embed_shape'], batch_size=hyperparams['batchsize'], name="node_embed_input")
        conv_2 = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='relu')(node_embed_input)
        flatten_z_mean = tf.keras.layers.Flatten()(conv_2)
        encode_dense_1 = tf.keras.layers.Dense(hyperparams['graph_hidden_dim'], activation='relu')(flatten_z_mean)
        encode_dense_2 = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], activation='relu')(encode_dense_1)
        self.graph_encoder = tf.keras.Model(inputs=node_embed_input, outputs=encode_dense_2, name='graph_encoder')
        #print(self.graph_encoder.summary())

        graph_decode_input = tf.keras.Input(shape=hyperparams['graph_embed_dim'], name='graph_embedding')
        decode_dense_1 = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], activation='relu')(graph_decode_input)
        repeat = tf.keras.layers.RepeatVector(hyperparams['feature_shape'][0])(decode_dense_1)
        gru_1 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(repeat)
        gru_2 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_1)
        gru_3 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_2)
        decode_dense_2 = tf.keras.layers.Dense(hyperparams['node_embed_shape'][-1], activation='relu')(gru_3)
        self.graph_decoder = tf.keras.Model(inputs=graph_decode_input, outputs=decode_dense_2, name='graph_decoder')
        #print(self.graph_decoder.summary())
        
        node_decode_input = tf.keras.Input(shape=hyperparams['node_embed_shape'], name='node_embedding')
        output_adj = GravityInspiredDecoder(\
                                node_encode_dim = hyperparams['node_embed_shape'][-1], lamb=hyperparams['lamb'], \
                                epsilon=hyperparams['epsilon'], normalize=hyperparams['normalize'], \
                                dropout=hyperparams['dropout'], act=tf.keras.activations.linear)(node_decode_input)
        output_feature = DenseFeatureDecoder(hyperparams['node_embed_shape'][-1], hyperparams['feature_shape'][1], \
                                    dropout=hyperparams['dropout'], act=tf.keras.activations.relu)(node_decode_input)
        self.node_decoder = tf.keras.Model(inputs=node_decode_input, outputs=[output_adj, output_feature], name="node_decoder")
        #print(self.node_decoder.summary())

        self._set_inputs(inputs=[feature_input,adj_input], outputs=[output_adj, output_feature])

        self.batch_size = hyperparams['batchsize']
        self.built = True

class GraphEmbedGravityGCNVAE(GraphEmbedGCNVAE):
    def __init__(self, **kwargs):
        super(GraphEmbedGravityGCNVAE, self).__init__(**kwargs)

    def build(self, hyperparams):
        feature_input = tf.keras.Input(shape=hyperparams['feature_shape'], batch_size=hyperparams['batchsize'], name="feature_input")
        adj_input = tf.keras.Input(shape=hyperparams['adj_shape'], batch_size=hyperparams['batchsize'], name="adj_input")
        conv_1 = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                num_features = hyperparams['feature_shape'][1],
                                node_encode_dim = hyperparams['node_hidden_dim'],
                                dropout = hyperparams['dropout'],
                                act = tf.nn.relu,
                                name='conv_1')([feature_input,adj_input])

        node_z_mean = GraphConvolution(num_node = hyperparams['feature_shape'][0],
                                    num_features = hyperparams['node_hidden_dim'],
                                    node_encode_dim = hyperparams['node_embed_shape'][-1],
                                    dropout = hyperparams['dropout'],
                                    act = tf.keras.activations.linear,
                                    name='node_z_mean')([conv_1,adj_input])
        self.node_encoder = tf.keras.Model(inputs=[feature_input,adj_input], outputs=node_z_mean, name='node_encoder')
        #print(self.node_encoder.summary())

        node_embed_input = tf.keras.Input(shape=hyperparams['node_embed_shape'], batch_size=hyperparams['batchsize'], name="node_embed_input")
        x = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='swish')(node_embed_input)
        x = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='swish')(x)
        x = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='swish')(x)
        x = tf.keras.layers.Flatten()(x)
        encode_dense_1 = tf.keras.layers.Dense(hyperparams['graph_hidden_dim'], activation='swish')(x)

        graph_z_mean = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], activation='swish', name="graph_z_mean")(encode_dense_1)
        graph_z_log_var = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], name="graph_z_log_var")(encode_dense_1)
        graph_z = Sampling(name="graph_z")([graph_z_mean, graph_z_log_var])

        self.graph_encoder = tf.keras.Model(inputs=node_embed_input, outputs=[graph_z_mean, graph_z_log_var, graph_z], name='graph_encoder')
        #print(self.graph_encoder.summary())

        graph_decode_input = tf.keras.Input(shape=hyperparams['graph_embed_dim'], name='graph_embedding')
        decode_dense_1 = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], activation='swish')(graph_decode_input)
        repeat = tf.keras.layers.RepeatVector(hyperparams['feature_shape'][0])(decode_dense_1)
        gru_1 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(repeat)
        gru_2 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_1)
        gru_3 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_2)
        decode_dense_2 = tf.keras.layers.Dense(hyperparams['node_embed_shape'][-1], activation='swish')(gru_3)
        self.graph_decoder = tf.keras.Model(inputs=graph_decode_input, outputs=decode_dense_2, name='graph_decoder')
        #print(self.graph_decoder.summary())
        
        node_decode_input = tf.keras.Input(shape=hyperparams['node_embed_shape'], name='node_embedding')
        output_adj = GravityInspiredDecoder(\
                                node_encode_dim = hyperparams['node_embed_shape'][-1], lamb=hyperparams['lamb'], \
                                epsilon=hyperparams['epsilon'], normalize=hyperparams['normalize'], \
                                dropout=hyperparams['dropout'], act=tf.keras.activations.linear)(node_decode_input)
        output_feature = DenseFeatureDecoder(hyperparams['node_embed_shape'][-1], hyperparams['feature_shape'][1], \
                                    dropout=hyperparams['dropout'], act=tf.keras.activations.relu)(node_decode_input)
        self.node_decoder = tf.keras.Model(inputs=node_decode_input, outputs=[output_adj, output_feature], name="node_decoder")
        #print(self.node_decoder.summary())

        self._set_inputs(inputs=[feature_input,adj_input], outputs=[output_adj, output_feature])

        self.num_node = hyperparams['feature_shape'][0]
        self.batch_size = hyperparams['batchsize']
        self.weight_ratio = hyperparams['weight_ratio']
        self.built = True