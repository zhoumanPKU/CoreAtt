#############说明：这个模型改动在于使用更常见的自注意力机制
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense

from ...feature_column import build_input_features, input_from_feature_columns
from ...layers.core import PredictionLayer, DNN
from ...layers.utils import combined_dnn_input
import tensorflow as tf
from tensorflow.keras.layers import Layer, Concatenate,Lambda
import tensorflow.keras.backend as K

def ple(inputs, boundaries):
    inputs = tf.reshape(inputs, [-1, 1])
    left = boundaries[:-1]
    right = boundaries[1:]
    boundaries = tf.convert_to_tensor(boundaries, dtype=tf.float32)
    boundaries = tf.reshape(boundaries, [1, -1])
    left = tf.convert_to_tensor(left, dtype=tf.float32)
    right = tf.convert_to_tensor(right, dtype=tf.float32)
    left = tf.reshape(left, [1, -1])
    right = tf.reshape(right, [1, -1])

    offset = (inputs - left) / (right - left)
    ones, zeros = tf.ones_like(offset), tf.zeros_like(offset)
    print(inputs.shape, boundaries.shape, ones.shape, zeros.shape)
    outputs = tf.where(inputs > right, ones, tf.where(inputs < left, zeros, offset))
    print(outputs)
    return outputs

class InformationFusionUnit(Layer):
    """信息融合单元 - 使用注意力机制融合多任务信息"""
    def __init__(self, hidden_dim, conc=0,T=1.0,noV=0,**kwargs):
        super(InformationFusionUnit, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.attention_weights = None
        self.conc=conc
        self.T=T
        self.noV=noV
    def build(self, input_shape):
        self.h1 = DNN([self.hidden_dim], activation='linear')
        self.h2 = DNN([self.hidden_dim], activation='linear')
        self.h3 = DNN([self.hidden_dim], activation='linear')
        super(InformationFusionUnit, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        # inputs是一个列表，包含来自不同任务的信息，有0个/1个/2个32维向量
        # 计算注意力权重
        if len(inputs)==0:
            return tf.zeros(self.hidden_dim)
        # 将输入列表转换为张量 (batch_size, num_inputs, input_dim)
        inputs_tensor = tf.stack(inputs, axis=1)
        batch_size = tf.shape(inputs_tensor)[0]
        num_inputs = tf.shape(inputs_tensor)[1]
        
        # 应用Q、K、V变换
        Q = self.h1(inputs_tensor)  # (batch_size, num_inputs, hidden_dim)
        K = self.h2(inputs_tensor)    # (batch_size, num_inputs, hidden_dim)
        V = self.h3(inputs_tensor)  # (batch_size, num_inputs, hidden_dim)
        if self.noV==1:
            V=inputs_tensor
        #V = inputs_tensor
        
        # 计算注意力分数: Q * K^T
        attention_scores = tf.matmul(Q, K, transpose_b=True)  # (batch_size, num_inputs, num_inputs)
        
        # 缩放注意力分数
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        
        attention_scores = attention_scores / tf.math.sqrt(dk)
        
        # 应用softmax获取注意力权重
        temper=self.T
        self.attention_weights = tf.nn.softmax((attention_scores)/temper, axis=-1)
        
        #self.attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # (batch_size, num_inputs, num_inputs)
        
        # 加权求和: 注意力权重 * V
        fused_output = tf.matmul(self.attention_weights, V)  # (batch_size, num_inputs, hidden_dim)
        
        
        fused_output = tf.reduce_mean(fused_output, axis=1)  # (batch_size, hidden_dim)



        return fused_output
    def get_attention_weights(self):
        """获取注意力权重张量"""
        return self.attention_weights
class SigmoidParamLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # 先创建无约束的变量（初始值可以是任意实数）
        self.raw_param = self.add_weight(
            shape=(),
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5),  # 初始值围绕0
            trainable=True,
            name="raw_param"
        )

    @property
    def zero_one_param(self):
        # 用sigmoid映射到(0,1)（可学习参数的实际取值）
        return tf.sigmoid(self.raw_param)

    def call(self, inputs):
        output = inputs * self.zero_one_param  # 使用映射后的0-1参数
        return output


def mymodel4(dnn_feature_columns, bottom_dnn_hidden_units=(256, 128), tower_dnn_hidden_units=(64,),
                 l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'),tasktype=0,ifu_hidden_dim=32):
    """Instantiates the SharedBottom multi-task learning Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param bottom_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared bottom DNN.
    :param tower_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks

    :return: A Keras model instance.
    """



    num_tasks = len(task_names)
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")

    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    shared_bottom_output = DNN(bottom_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(
        dnn_input)
    t_output = []
    tasks_output = []

    if 1:
        boundary=[]
        '''
        boundary.append([0.0037130112759768963,0.11951162442564965,0.178619621694088,0.224365109950304,0.26372631192207335,0.2985670939087868,0.3312439382076263,0.3625532016158104,0.392257297039032,0.4200046002864838,0.4457378685474396,0.46958956122398376,0.4943672835826873,0.5222922205924988,0.5527025759220123,0.5848515182733536,0.619264554977417,0.6578819304704666,0.7026961028575899,0.7596274584531784,0.9662696123123169])
        boundary.append([4.268302654963918e-05,0.004436898673884571,0.014079219661653042,0.024230319634079924,0.03586941733956337,0.04958068300038576,0.06514997109770775,0.08321793414652347,0.1040564239025116,0.1276190593838692,0.15396616607904434,0.18229031115770342,0.21143953800201415,0.2425158642232418,0.29074913561344146,0.35176537185907364,0.41460854411125186,0.47078593671321867,0.5359052002429963,0.6218150824308395,0.9245381951332092])
        boundary.append([0.00026476033963263035,0.0014535810565575956,0.0018188934423960744,0.002104118699207902,0.0023557132109999655,0.0025895614526234567,0.002815540693700313,0.0030465306830592454,0.0032870241906493903,0.003550969320349395,0.003848759224638343,0.004202782013453543,0.0046596479602158064,0.005254882667213681,0.006099148048087953,0.007370776729658246,0.009459304623305798,0.013612475525587796,0.024216620810329936,0.07270290516316887,0.8379064202308655])
        boundary.append([0.00020952516933903098,0.0005852549045812338,0.0006845342402812093,0.0007626504375366494,0.0008320511435158551,0.0008993870724225417,0.0009668772749137133,0.0010347876232117414,0.0011072627734392882,0.0011855782649945468,0.0012727587018162012,0.0013680315285455435,0.0014792253961786626,0.0016070511948782955,0.0017561402753926813,0.0019409600645303726,0.002182543091475964,0.002526086114812641,0.003090511565096683,0.004335063020698724,0.0693494901061058])
        boundary.append([0.0005813157185912132,0.0008030412544030696,0.0008656018704641611,0.000946548132924363,0.0010092739015817643,0.001079227979062125,0.0011586343171074986,0.0012398042890708894,0.0013309927424415946,0.0014346617506816983,0.0015538394800387323,0.0016945649927947672,0.0018685868475586172,0.002098013914655894,0.0024078656919300553,0.002827585267368704,0.003440921101719142,0.00441512116231024,0.0061429812107235224,0.010739530622959136,0.25290098786354065])
        '''
        boundary.append([3.130421921682114e-09,0.7925992608070374,0.8444417715072632,0.8605466485023499,0.870436429977417,0.8771074116230011,0.8837410807609558,0.8893489301204681,0.8941320896148682,0.8988560438156128,0.9032672047615051,0.907733142375946,0.9130207419395446,0.9193947315216064,0.930152177810669,0.9460394084453583,0.9571723818778992,0.9626244723796844,0.9668577194213868,0.9748817443847656,0.9907673001289368])
        boundary.append([0.03288131207227707,0.9295856714248657,0.9496575951576233,0.9578750729560852,0.9631479382514954,0.9669541716575623,0.9695847630500793,0.9715658962726593,0.973465895652771,0.9750070154666901,0.9764947891235352,0.9778939485549927,0.9793342351913452,0.9808031558990479,0.982381546497345,0.9841803908348083,0.9860219955444336,0.9877190232276917,0.9892191290855408,0.9907516479492188,0.9963548183441162])
        boundary.append([0.0004006506351288408,0.004393391264602542,0.007036286871880294,0.009967478923499584,0.013541192375123502,0.017978740856051445,0.023827586323022842,0.031412498280405994,0.04001982137560845,0.049390994012355804,0.05805225297808647,0.06615186482667923,0.07385128736495972,0.08154964447021484,0.08926199376583097,0.09755508601665497,0.10605533421039581,0.11590824276208878,0.1293051689863205,0.1576520577073096,0.7942579984664917])
        boundary.append([0.00035233073867857456,0.0026246328838169576,0.0036100163590162995,0.004649267066270113,0.005864315200597049,0.0073884171433746815,0.008866584859788418,0.010572086647152899,0.012146110460162163,0.013685680367052555,0.015164574608206749,0.016676679253578186,0.018329843878746033,0.019782818853855133,0.021238310262560844,0.023141877725720406,0.025000285357236862,0.026658952236175537,0.030296102166175842,0.03750969842076292,0.6064860224723816])
        boundary.append([0.004069695249199867,0.009134511929005384,0.010292265005409718,0.010943857673555612,0.011477669700980186,0.012001152150332928,0.01254098117351532,0.013155167549848556,0.013758131302893162,0.014348269160836936,0.014968843199312687,0.01562681421637535,0.016322173178195953,0.0170286912471056,0.01789412386715412,0.01895969547331333,0.01983639970421791,0.020675189793109894,0.022155512124300003,0.0262319203466177,0.046798478811979294])
        boundary_maps = {}
        boundary_maps['click_target'] = boundary[0]
        boundary_maps['long_view'] = boundary[1]
        boundary_maps['like_target'] = boundary[2]
        boundary_maps['forward_target'] = boundary[3]
        boundary_maps['follow_target'] = boundary[4]


        t_output2=[]
        num_tasks = len(task_types)
        sigmoid_layers = [SigmoidParamLayer() for _ in range(num_tasks)] 
        for i, (task_type, task_name) in enumerate(zip(task_types, task_names)):
        
            sbtemp=shared_bottom_output
            ifu = InformationFusionUnit(32,0,0.8,1)
            fused_output = ifu(t_output)
            # if i!=0 and i!=2 and i!=3:
            if i==num_tasks-1:
                #最后一个目标
                sbtemp=Concatenate()([sbtemp,(1-sigmoid_layers[i].zero_one_param)*fused_output+sigmoid_layers[i].zero_one_param*(tf.reduce_mean(t_output2, axis=0))])
            tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                           name='tower_' + task_name)(sbtemp)
        
            tower_output_nog=tf.stop_gradient(tower_output)
            t_output.append(tower_output_nog)
    
            logit = Dense(1, use_bias=False)(tower_output)
            output = PredictionLayer(task_type, name=task_name)(logit)
            tasks_output.append(output)
            output_nog=tf.stop_gradient(output)
            if i<num_tasks-2:
                # le=ple(output_nog,boundary[i]) 
                le=ple(output_nog,boundary_maps[task_name])
                batch_sums = tf.reduce_sum(le, axis=1)  # 结果形状：[batch_size]

                # 第二步：增加一个维度，将形状转换为[batch_size, 1]
                w = tf.expand_dims(batch_sums, axis=1)
                w=w/20
                t_output2.append((1+w)*tf.stop_gradient(tower_output))#(1-output_nog)*
    
    model = Model(inputs=inputs_list, outputs=tasks_output)
    return model
