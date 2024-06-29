from layers import *
import tensorflow as tf
from graph_embed_model import *

class ConditionalGravityGCNVAE(GraphEmbedGCNVAE):
    def __init__(self, **kwargs):
        super(ConditionalGravityGCNVAE, self).__init__(**kwargs)

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
        onehot_condition_input = tf.keras.Input(shape=(1,), batch_size=hyperparams['batchsize'], name="onehot_condition_input")
        concat_enc_input = ConcatEncoderInput()([node_embed_input,onehot_condition_input])
        x = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='swish')(concat_enc_input)
        x = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='swish')(x)
        x = tf.keras.layers.Conv1D(hyperparams['graph_conv_filters'], hyperparams['graph_conv_kernel'], activation='swish')(x)
        x = tf.keras.layers.Flatten()(x)
        encode_dense_1 = tf.keras.layers.Dense(hyperparams['graph_hidden_dim'], activation='swish')(x)

        graph_z_mean = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], activation='swish', name="graph_z_mean")(encode_dense_1)
        graph_z_log_var = tf.keras.layers.Dense(hyperparams['graph_embed_dim'], name="graph_z_log_var")(encode_dense_1)
        graph_z = Sampling(name="graph_z")([graph_z_mean, graph_z_log_var])

        self.graph_encoder = tf.keras.Model(inputs=[node_embed_input,onehot_condition_input], outputs=[graph_z_mean, graph_z_log_var, graph_z], name='graph_encoder')
        #print(self.graph_encoder.summary())

        graph_decode_input = tf.keras.Input(shape=(hyperparams['graph_embed_dim'],), batch_size=hyperparams['batchsize'], name='graph_embedding')
        condition_input = tf.keras.Input(shape=(1,), batch_size=hyperparams['batchsize'], name='condition')
        concat_dec_input = ConcatDecoderInput()([graph_decode_input, condition_input])
        decode_dense_1 = tf.keras.layers.Dense(2*hyperparams['graph_embed_dim'], activation='swish')(concat_dec_input)
        repeat = tf.keras.layers.RepeatVector(hyperparams['feature_shape'][0])(decode_dense_1)
        gru_1 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(repeat)
        gru_2 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_1)
        gru_3 = tf.keras.layers.GRU(hyperparams['graph_gru_dim'], return_sequences = True)(gru_2)
        decode_dense_2 = tf.keras.layers.Dense(hyperparams['node_embed_shape'][-1], activation='swish')(gru_3)
        self.graph_decoder = tf.keras.Model(inputs=[graph_decode_input,condition_input], outputs=decode_dense_2, name='graph_decoder')
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

    def call(self, inputs, *args, **kwargs):
        data, c = inputs
        x, _ = data

        node_z_mean = self.node_encoder(x)
        z_mean, z_log_std, z = self.graph_encoder([node_z_mean, c])

        graph_reconstructed = self.graph_decoder([z,c])
        adj, feature = self.node_decoder(graph_reconstructed)
        return adj, feature

    def encode(self, inputs):
        data, c = inputs
        x, _ = data
        node_z_mean = self.node_encoder.predict(x, batch_size=self.batch_size)
        z_mean, z_log_std, z = self.graph_encoder.predict([node_z_mean,c], batch_size=self.batch_size)
        return z

    def decode(self, z, c):
        graph_reconstructed = self.graph_decoder([z,c])
        adj, feature = self.node_decoder.predict(graph_reconstructed)
        return adj, feature

    @tf.function
    def train_step(self, inputs):
        data, c = inputs
        x, y = data
        fmx, _ = x
        target, norm, pos_weight = y

        with tf.GradientTape() as tape:
            node_z_mean = self.node_encoder(x)
            z_mean, z_log_std, z = self.graph_encoder([node_z_mean, c])

            graph_reconstructed = self.graph_decoder([z,c])
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
        grads = [tf.clip_by_value(g, -1, 1) for g in grads]
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
        data, c = inputs
        x, y = data
        fmx, _ = x
        target, norm, pos_weight = y

        node_z_mean = self.node_encoder(x)
        z_mean, z_log_std, z = self.graph_encoder([node_z_mean, c])

        graph_reconstructed = self.graph_decoder([z,c])
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