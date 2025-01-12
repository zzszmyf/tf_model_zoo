"""
wide_deep_multi_optimizer.py

A Wide & Deep model implementation with multiple optimizers:
- Adagrad for embedding layers (suitable for sparse features)
- FTRL for wide part (good for linear models with sparse features)
- Adam for deep part (effective for deep neural networks)

Architecture:
- Wide: Linear model running on CPU with FTRL
- Deep: DNN with embeddings running on GPU
- Embedding: Categorical feature embedding with Adagrad

Author: zzszmyf
Date: 2025-01-13
Email: zzszmyf@outlook.com
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 版本兼容设置
if tf.__version__.startswith('1.'):
    tf.enable_eager_execution()
    AUTO_TUNE = 1000
    tf_version = 1
else:
    AUTO_TUNE = tf.data.AUTOTUNE
    tf_version = 2

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.numeric_features = ['age', 'income']
        self.categorical_features = ['occupation', 'education', 'gender']
        self.feature_dims = {}
        self.numeric_means = {}
        self.numeric_stds = {}

    def fit(self, df):
        # 处理类别特征
        for feat in self.categorical_features:
            le = LabelEncoder()
            df[feat] = le.fit_transform(df[feat])
            self.label_encoders[feat] = le
            self.feature_dims[feat] = len(le.classes_)

        # 处理数值特征
        for feat in self.numeric_features:
            self.numeric_means[feat] = df[feat].mean()
            self.numeric_stds[feat] = df[feat].std()
            df[feat] = (df[feat] - self.numeric_means[feat]) / self.numeric_stds[feat]
        return df

    def transform(self, df):
        df = df.copy()
        # 处理类别特征
        for feat in self.categorical_features:
            le = self.label_encoders[feat]
            df[feat] = le.transform(df[feat])

        # 处理数值特征
        for feat in self.numeric_features:
            df[feat] = (df[feat] - self.numeric_means[feat]) / self.numeric_stds[feat]
        return df

class InputPipeline:
    def __init__(self, numeric_features, categorical_features, batch_size=1024):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.batch_size = batch_size

    def create_dataset(self, df, labels, shuffle=True):
        features_dict = {}
        # 处理数值特征
        for feat in self.numeric_features:
            values = df[feat].values
            features_dict[feat] = values.astype(np.float32)

        # 处理类别特征
        for feat in self.categorical_features:
            values = df[feat].values
            features_dict[feat] = values.astype(np.int32)

        labels = labels.reshape(-1, 1).astype(np.float32)

        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((features_dict, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(df))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTO_TUNE)
        return dataset

class DeepLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units, dropout_rates):
        super().__init__()
        self.dense_layers = []
        self.dropout_layers = []
        self.batch_norm_layers = []

        for units, dropout_rate in zip(hidden_units, dropout_rates):
            self.dense_layers.append(tf.keras.layers.Dense(
                units,
                activation=None,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            ))
            self.batch_norm_layers.append(tf.keras.layers.BatchNormalization())
            self.dropout_layers.append(tf.keras.layers.Dropout(dropout_rate))

    def call(self, inputs, training=None):
        x = inputs
        for dense, batch_norm, dropout in zip(
            self.dense_layers,
            self.batch_norm_layers,
            self.dropout_layers
        ):
            x = dense(x)
            x = batch_norm(x, training=training)
            x = tf.nn.relu(x)
            x = dropout(x, training=training)
        return x

class WideDeepModel(tf.keras.Model):
    def __init__(self, feature_dims, numeric_features, embedding_dim=16):
        super().__init__()
        self.feature_dims = feature_dims
        self.numeric_features = numeric_features
        self.embedding_dim = embedding_dim

        # Embedding layers
        self.embeddings = {}
        for feat, dim in feature_dims.items():
            self.embeddings[feat] = tf.keras.layers.Embedding(
                dim,
                embedding_dim,
                embeddings_regularizer=tf.keras.regularizers.l2(1e-4),
                name=f'embedding_{feat}'
            )

        # Deep part
        self.deep = DeepLayer(
            hidden_units=[128, 64, 32],
            dropout_rates=[0.2, 0.2, 0.2]
        )

        # Wide part
        total_input_dim = len(numeric_features) + len(feature_dims) * embedding_dim
        with tf.device('/cpu:0'):
            self.wide = tf.keras.layers.Dense(1, use_bias=True)

        # Final output
        self.final_dense = tf.keras.layers.Dense(1, activation='sigmoid')

        # Initialize optimizers based on TF version
        if tf_version == 1:
            self.embedding_optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
            with tf.device('/cpu:0'):
                self.wide_optimizer = tf.train.FtrlOptimizer(learning_rate=0.01)
            self.deep_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        else:
            self.embedding_optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
            with tf.device('/cpu:0'):
                self.wide_optimizer = tf.keras.optimizers.Ftrl(learning_rate=0.01)
            self.deep_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def get_embedding_variables(self):
        embedding_vars = []
        for layer in self.embeddings.values():
            embedding_vars.extend(layer.trainable_variables)
        return embedding_vars

    def get_wide_variables(self):
        return self.wide.trainable_variables

    def get_deep_variables(self):
        deep_vars = []
        deep_vars.extend(self.deep.trainable_variables)
        deep_vars.extend(self.final_dense.trainable_variables)
        return deep_vars

    def call(self, inputs, training=None):
        # Process numeric features
        numeric_inputs = []
        for feat in self.numeric_features:
            x = tf.cast(inputs[feat], tf.float32)
            if len(x.shape) == 1:
                x = tf.expand_dims(x, -1)
            numeric_inputs.append(x)
        numeric_concat = tf.concat(numeric_inputs, axis=1)

        # Process categorical features
        embedding_outputs = []
        for feat in self.feature_dims.keys():
            feat_input = tf.cast(inputs[feat], tf.int32)
            if len(feat_input.shape) == 1:
                feat_input = tf.expand_dims(feat_input, -1)
            embed = self.embeddings[feat](feat_input)
            embed_flat = tf.reshape(embed, [-1, self.embedding_dim])
            embedding_outputs.append(embed_flat)

        # 合并所有 embedding 输出
        if embedding_outputs:
            embed_concat = tf.concat(embedding_outputs, axis=1)
        else:
            embed_concat = tf.zeros([tf.shape(numeric_concat)[0], 0])

        # 合并数值特征和 embedding 特征
        deep_input = tf.concat([numeric_concat, embed_concat], axis=1)

        # Deep path
        deep_out = self.deep(deep_input, training=training)

        # Wide path
        with tf.device('/cpu:0'):
            wide_out = self.wide(deep_input)

        # 合并 wide 和 deep 输出
        combined = tf.concat([deep_out, wide_out], axis=1)
        return self.final_dense(combined)

@tf.function
def train_step(model, inputs, labels):
    if tf_version == 1:
        with tf.GradientTape(persistent=True) as tape:
            tape._persistent = True

            # Forward pass
            predictions = model(inputs, training=True)

            # Ensure labels are float32
            labels = tf.cast(labels, tf.float32)
            if len(labels.shape) == 1:
                labels = tf.expand_dims(labels, -1)

            # Calculate loss
            loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(labels, predictions)
            )

            # Add regularization losses
            if model.losses:
                loss += tf.add_n(model.losses)

        # Calculate gradients
        embedding_vars = model.get_embedding_variables()
        with tf.device('/cpu:0'):
            wide_vars = model.get_wide_variables()
        deep_vars = model.get_deep_variables()

        if embedding_vars:
            embedding_grads = tape.gradient(loss, embedding_vars)
            model.embedding_optimizer.apply_gradients(zip(embedding_grads, embedding_vars))

        if wide_vars:
            wide_grads = tape.gradient(loss, wide_vars)
            model.wide_optimizer.apply_gradients(zip(wide_grads, wide_vars))

        if deep_vars:
            deep_grads = tape.gradient(loss, deep_vars)
            model.deep_optimizer.apply_gradients(zip(deep_grads, deep_vars))

        del tape
        return loss, predictions
    else:
        with tf.GradientTape(persistent=True) as tape:
            predictions = model(inputs, training=True)
            labels = tf.cast(labels, tf.float32)
            if len(labels.shape) == 1:
                labels = tf.expand_dims(labels, -1)
            loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(labels, predictions)
            )
            if model.losses:
                loss += tf.add_n(model.losses)

        embedding_vars = model.get_embedding_variables()
        with tf.device('/cpu:0'):
            wide_vars = model.get_wide_variables()
        deep_vars = model.get_deep_variables()

        if embedding_vars:
            embedding_grads = tape.gradient(loss, embedding_vars)
            model.embedding_optimizer.apply_gradients(zip(embedding_grads, embedding_vars))
        with tf.device('/cpu:0'):
            if wide_vars:
                wide_grads = tape.gradient(loss, wide_vars)
                model.wide_optimizer.apply_gradients(zip(wide_grads, wide_vars))

        if deep_vars:
            deep_grads = tape.gradient(loss, deep_vars)
            model.deep_optimizer.apply_gradients(zip(deep_grads, deep_vars))

        del tape
        return loss, predictions

def evaluate_model(model, dataset):
    losses = []
    predictions = []
    labels_all = []

    for batch in dataset:
        inputs, batch_labels = batch
        if len(batch_labels.shape) == 1:
            batch_labels = tf.expand_dims(batch_labels, -1)

        batch_predictions = model(inputs, training=False)
        loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(batch_labels, batch_predictions)
        )

        losses.append(float(loss))
        predictions.extend(batch_predictions.numpy().flatten())
        labels_all.extend(batch_labels.numpy().flatten())

    # 计算 AUC
    auc = roc_auc_score(labels_all, predictions)
    return np.mean(losses), auc, predictions, labels_all

def generate_synthetic_data(n_samples=10000):
    np.random.seed(42)

    # 生成特征
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'occupation': np.random.choice(['engineer', 'doctor', 'teacher', 'lawyer', 'artist'], n_samples),
        'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], n_samples),
        'gender': np.random.choice(['male', 'female'], n_samples)
    }

    # 生成标签 (模拟点击率)
    age_factor = (data['age'] - 35) / 10
    income_factor = (data['income'] - 50000) / 20000

    occupation_effect = {
        'engineer': 0.3,
        'doctor': 0.4,
        'teacher': 0.2,
        'lawyer': 0.35,
        'artist': 0.25
    }

    education_effect = {
        'high_school': 0.2,
        'bachelor': 0.3,
        'master': 0.4,
        'phd': 0.5
    }

    gender_effect = {'male': 0.3, 'female': 0.3}

    # 计算点击概率
    probs = np.zeros(n_samples)
    for i in range(n_samples):
        prob = 0.3  # 基础点击率
        prob += 0.1 * age_factor[i]
        prob += 0.1 * income_factor[i]
        prob += occupation_effect[data['occupation'][i]]
        prob += education_effect[data['education'][i]]
        prob += gender_effect[data['gender'][i]]
        probs[i] = max(0.01, min(0.99, prob))

    # 生成实际点击标签
    labels = np.random.binomial(1, probs)

    return pd.DataFrame(data), labels

def main():
    # 生成合成数据
    df, labels = generate_synthetic_data(n_samples=100000)

    # 数据预处理
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit(df)

    # 划分训练集和测试集
    train_df, test_df, train_labels, test_labels = train_test_split(
        df_processed,
        labels,
        test_size=0.2,
        random_state=42
    )

    # 创建数据管道
    pipeline = InputPipeline(
        preprocessor.numeric_features,
        preprocessor.categorical_features,
        batch_size=1024
    )

    train_dataset = pipeline.create_dataset(train_df, train_labels)
    test_dataset = pipeline.create_dataset(test_df, test_labels, shuffle=False)

    # 创建模型
    model = WideDeepModel(
        preprocessor.feature_dims,
        preprocessor.numeric_features,
        embedding_dim=128
    )

    # 训练循环
    best_auc = 0
    patience = 5
    patience_counter = 0
    epochs = 50

    for epoch in range(epochs):
        epoch_losses = []
        try:
            for batch in train_dataset:
                inputs, batch_labels = batch
                loss, predictions = train_step(model, inputs, batch_labels)
                epoch_losses.append(float(loss))

            # 评估
            test_loss, test_auc, _, _ = evaluate_model(model, test_dataset)

            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Train Loss: {np.mean(epoch_losses):.4f}')
            print(f'Test Loss: {test_loss:.4f}')
            print(f'Test AUC: {test_auc:.4f}')
            print('-------------------------')

            # 早停检查
            if test_auc > best_auc:
                best_auc = test_auc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        except Exception as e:
            print(f"Error in epoch {epoch + 1}: {str(e)}")
            continue
