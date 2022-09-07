"""Transformer 구현

논문 제목: Attention is All you Need

encoder-decoder 모델을 self-attention과 cross-attention으로 구현
input의 attention을 구할 때 matmul을 사용하여 전체를 계산 상에 이어서 기존 RNN 모델의 정보 병목 현상 방지

요약: 이런이런 attention을 고려하여 중요한 애들이 이런 식으로 배치됐을 때 다음으로 나와야 하는 단어는 무엇인가? 학습

ToDO:
    * mask 처럼 검증이 필요한 부분은 따로 함수로 빼서 확인
"""
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import metrics, losses, optimizers, layers


def positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    """
    tensor에 주입할 위치 정보 반환

    Args:
        seq_len: sequence length
        d_model: embedding layer output dim

    Returns:
        positional information which has shape (seq_len, d_model)
    """
    # (seq #, embedding #)
    positional_information = np.ones(shape=(seq_len, d_model), dtype=np.float32)
    pos_idx = np.arange(1, seq_len + 1, dtype=np.float32).reshape(-1, 1)  # (seq_len, 1)
    dim_idx = np.arange(1, d_model + 1, dtype=np.float32).reshape(1, -1)  # (1, d_model)
    divisor = np.power(10000, (2 * dim_idx) / d_model)  # (1, d_model)

    # PE(pos_idx, dim_idx) = pos_idx / pow(10000, 2 * dim_idx / d_model)
    positional_information = (pos_idx * positional_information) / divisor  # (seq #, embedding #)

    output = []
    for dim_idx in range(d_model):
        if dim_idx % 2 == 0:
            output.append(tf.sin(positional_information[:, dim_idx]))
        else:
            output.append(tf.cos(positional_information[:, dim_idx]))

    positional_information = tf.stack(output, axis=1)
    return positional_information


def scaled_dot_product_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask_start_index: int = None) -> tf.Tensor:
    """
    q * k 로 attention 구하고 v에 곱해줘서 attention 고려 => self-attention

    Args:
        q: tensor data which has shape(batch #, seq_len, d_model // h)
        k: tensor data which has shape(batch #, seq_len, d_model // h)
        v: tensor data which has shape(batch #, seq_len, d_model // h)
        mask_start_index: if this is not None, future information of sequence starting from mask_start_index
                          will be blocked
    Returns:
        softmax(Q * K_t / root(d_k)) * V which has shape (batch #, seq_len, d_model // h)
    """
    k_t = tf.transpose(k, perm=[0, 2, 1])  # (d_model // h, seq_len)
    # sequence 끼리의 유사도 == attention => 확률 형태로 attention 표현
    tmp_out = tf.matmul(q, k_t) / tf.sqrt(tf.cast(k.shape[2], tf.float32))  # (seq_len, seq_len)
    # Masking (opt.)
    if mask_start_index is not None:
        mask = np.zeros(shape=(tmp_out.shape[1], tmp_out.shape[2]))
        mask[:, mask_start_index:] = -1e9
        tmp_out += mask  # 매우 작은 값으로 만들어 softmax 후 0이 되도록 하여 masking 수행

    tmp_out = tf.nn.softmax(tmp_out)
    out = tf.matmul(tmp_out, v)  # (batch #, seq_len, d_model // h)
    return out


def multi_head_attention(h: int, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, mask_start_index: int = None) -> tf.Tensor:
    """
    여러가지 관점에서 attention을 본다.

    attention(query, key) * value => value에서 집중할 부분에 weight 줌

    ** cross-attention 시에 encoder (k, v)와 decoder (q)의 길이가 서로 다를 수 있다.

    Args:
        h: number of heads
        query: tensor data which has shape (batch #, seq_len, d_model)
        key: tensor data which has shape (batch #, seq_len, d_model)
        value: tensor data which has shape (batch #, seq_len, d_model)
        mask_start_index: if this is not None, future information of sequence starting from mask_start_index
                          will be blocked

    Returns:
        MultiHead(Q, K, V) = FC(Concat(head_1, ..., head_h)) which has shape (batch #, seq_len, d_model)
    """
    def head_attention(out_dim: int, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor):
        # projection
        q = layers.Dense(out_dim, activation='relu')(q)
        k = layers.Dense(out_dim, activation='relu')(k)
        v = layers.Dense(out_dim, activation='relu')(v)
        return scaled_dot_product_attention(q, k, v, mask_start_index)

    head_dim = query.shape[2] // h  # head_dim = d_model // h => cross-attention 시에 주의

    heads = []
    for _ in range(h):
        heads.append(head_attention(head_dim, tf.identity(query), tf.identity(key), tf.identity(value)))
    heads = tf.concat(heads, axis=2)
    out = layers.Dense(head_dim * h, activation='relu')(heads)
    return out


def encoder_layer(input_tensor: tf.Tensor) -> tf.Tensor:
    """
    Args:
        input_tensor: encoder's input data which has shape (batch #, seq_len, d_model)
    Returns:
    """
    # self-attention
    multi_head_out = multi_head_attention(h=8, query=tf.identity(input_tensor), key=tf.identity(input_tensor), value=tf.identity(input_tensor))
    add_norm_1 = layers.LayerNormalization()(input_tensor + multi_head_out)  # relu 적용해야하나?

    ffn = layers.Dense(input_tensor.shape[2] * 4, activation='relu')(add_norm_1)
    ffn = layers.Dense(input_tensor.shape[2])(ffn)  # (batch #, seq_len, d_model)

    add_norm_2 = layers.LayerNormalization()(ffn + add_norm_1)
    return add_norm_2


def decoder_layer(input_tensor: tf.Tensor, enc_out: tf.Tensor, mask_start_index: int) -> tf.Tensor:
    """
    Args:
        input_tensor: decoder's input data which has shape (batch #, seq_len, d_model)
        enc_out: encoder output which has shape (batch #, seq_len, d_model)
        mask_start_index: future information of sequence starting from mask_start_index
                          will be blocked
    Returns:
        tensor data which has shape (batch #, seq_len, d_model)
    """
    # Masked Multi-Head Attention으로 future information이 유출되지 않도록 함
    # decoder input은 <Start> 등의 token으로 시작한다고 생각함
    dec_self_attention = multi_head_attention(h=8, query=tf.identity(input_tensor), key=tf.identity(input_tensor),
                                              value=tf.identity(input_tensor), mask_start_index=mask_start_index)
    add_norm1 = layers.LayerNormalization()(dec_self_attention + input_tensor)
    cross_attention = multi_head_attention(h=8, query=tf.identity(add_norm1), key=tf.identity(enc_out),
                                           value=tf.identity(enc_out))
    add_norm2 = layers.LayerNormalization()(cross_attention + add_norm1)

    ffn = layers.Dense(input_tensor.shape[2] * 4, activation='relu')(add_norm2)
    ffn = layers.Dense(input_tensor.shape[2])(ffn)  # (batch #, seq_len, d_model)

    add_norm3 = layers.LayerNormalization()(add_norm2 + ffn)  # (batch #, seq_len, d_model)

    return add_norm3


def transformer(seq_len: int = 150, feat_len: int = 1, d_model: int = 512, vocab_size: int = 2000, pos_enc: tf.Tensor = None) -> keras.Model:
    """

    Args:
        seq_len: input sequence 길이 (pad_sequences 등으로 맞춰줄 것)
        feat_len: 하나의 sequence element가 가지는 feature 개수 (언어일 경우 단어 하나가 feature이므로 1 기대)
        d_model: embedding 후 dimension 크기 (논문에선 512 사용)
        vocab_size: token integer의 최대값 + 1, 영어인 경우 2000 사용
        pos_enc: embedding 이후 더해질 positional encoding 값
    Returns:
        transformer 모델

    """
    # Encoder part
    input_tensor = layers.Input(shape=(seq_len, feat_len))  # (batch #, seq_len, feat_len)
    enc_out = input_tensor
    # (batch #, seq #, feat #, d_model)
    # enc_input = layers.Embedding(input_dim=vocab_size, output_dim=d_model, input_length=seq_len)(input_tensor)
    # enc_input = tf.squeeze(enc_input, axis=2)  # NLP feat # == 1 가정, shape (batch #, seq_len, d_model)
    # enc_input = enc_input + pos_enc  # (batch #, seq_len, d_model)
    # enc_out = enc_input
    # for _ in range(6):  # stack of encoder layer (N=6 in paper)
    #     enc_out = encoder_layer(enc_out)

    # Decoder part
    output_tensor = layers.Input(shape=(seq_len, feat_len))  # (batch #, seq_len, feat_len)
    # (batch #, seq #, feat #, d_model)
    dec_input = layers.Embedding(input_dim=vocab_size, output_dim=d_model, input_length=seq_len)(output_tensor)
    dec_input = tf.squeeze(dec_input, axis=2)  # NLP feat # == 1 가정, shape (batch #, seq_len, d_model)
    dec_input = dec_input + pos_enc  # (batch #, seq_len, d_model)
    pred_vocabs = []
    for seq_idx in range(seq_len):
        dec_out = dec_input
        for _ in range(6):  # stack of decoder (N=6 in paper)
            dec_out = decoder_layer(dec_out, enc_out, seq_idx + 1)  # (batch #, seq_len, d_model)
        word_pred = layers.Dense(vocab_size)(dec_out)  # (batch #, seq_len, vocab_size)
        pred_vocabs.append(tf.argmax(word_pred[:, seq_idx, :], axis=1))  # 현재 sequence index에 대한 예측을 뽑는 것으로 생각

    return keras.Model(inputs=[input_tensor, output_tensor], outputs=[enc_out, pred_vocabs])


if __name__ == "__main__":
    seq_len = 5
    d_model = 512
    pos_enc = positional_encoding(seq_len, d_model)
    model = transformer(seq_len=seq_len, d_model=d_model, pos_enc=pos_enc)
    rand_input = tf.random.normal((1, seq_len, 1))
    rand_output = tf.random.normal((1, seq_len, 1))
    inputs = (rand_input, rand_output)
    # print(model.summary())
    out = model(inputs)
    print(out[1])
    print(tf.shape(out[1]))

