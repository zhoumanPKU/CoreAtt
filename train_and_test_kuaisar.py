
import sys
import os
import csv
import argparse
import fcntl
import time
import json
import random
from multiprocessing import Process
# 获取上级目录路径（parent_dir）
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将目标文件夹路径添加到sys.path
target_folder = os.path.join(parent_dir, 'DeepCTR')
sys.path.insert(0, target_folder)

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names,VarLenSparseFeat

from deepctr.models import mymodel4
#rom deepctr.utils import check_model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
#import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback,EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE
from tensorflow.keras.models import load_model
import uuid
import datetime
from sklearn.metrics import roc_auc_score
import numpy as np
import math
def create_triangular_table(results, metric_name, task_names):
    """创建三角形表格，行和列均为任务，仅填充上三角部分"""
    # 初始化表格
    table = pd.DataFrame(
        index=task_names, 
        columns=task_names
    )
    
    # 填充表格
    for key, values in results.items():
        i, j = eval(key)  # 解析键"(i,j)"为索引
        if i < j:  # 只填充上三角
            # 将5个桶的结果格式化为字符串
            formatted_values = [f"{v:.4f}" for v in values]
            table.iloc[i, j] = " | ".join(formatted_values)
    
    return table
def binary_entropy(prob):
    """计算二元数据的熵（0-1分布）"""
    if prob == 0 or prob == 1:
        return 0.0
    return -prob * math.log2(prob) - (1 - prob) * math.log2(1 - prob)
def calculate_group_metrics(task_index, pred_values, true_labels, task_names):
    """
    对当前任务的预估值从大到小排序，平均分为5组，计算每组中后续任务真实值的
    方差、概率（1的比例）和熵
    
    参数：
    - task_index：当前任务索引（0-4）
    - pred_values：当前任务的预估值数组（shape=(N,)）
    - true_labels：字典，{任务名: 真实值数组}
    - task_names：任务名称列表
    
    返回：
    - 字典：包含三个键（variances, probabilities, entropies），每个键对应的值
             是一个字典，键为"(i,j)"表示从任务i到任务j，值为5个桶的结果列表
    """

    results = {
        "variances": {},
        "probabilities": {},
        "entropies": {}
    }
    n_samples = len(pred_values)
    
    # 1. 对预估值从大到小排序，获取排序索引（降序）
    sorted_indices = np.argsort(pred_values.reshape(-1))[::-1]
    
    # 2. 计算每组的样本量（尽可能平均分配）
    group_size = n_samples // 5  # 基础组大小
    remainder = n_samples % 5    # 剩余样本（前remainder组各多1个）
    
    # 3. 定义每组的索引范围（左闭右开）
    group_ranges = []
    start = 0
    for i in range(5):
        end = start + group_size + (1 if i < remainder else 0)
        group_ranges.append((start, end))
        start = end
    
    # 4. 遍历每个组，计算后续任务的三个指标
    for group_id in range(5):
        # 获取当前组的样本索引（基于排序后的索引）
        start_idx, end_idx = group_ranges[group_id]
        group_sample_indices = sorted_indices[start_idx:end_idx]
        # 遍历后续任务（j > 当前任务索引）
        for j in range(task_index + 1, 5):
            # 获取后续任务j的真实值（仅当前组的样本）
            subsequent_true = true_labels[task_names[j]][group_sample_indices]
            n = len(subsequent_true)
            
            # 计算概率（1的比例）
            prob = np.sum(subsequent_true) / n if n > 0 else 0.0
            
            
            # 计算方差（总体方差）
            variance = np.var(subsequent_true, ddof=0) if n > 0 else 0.0
            
            # 计算熵
            entropy = binary_entropy(prob)
            
            # 构建键名
            key = f"({task_index},{j})"
            
            # 初始化列表（如果不存在）
            for metric in ["variances", "probabilities", "entropies"]:
                if key not in results[metric]:
                    results[metric][key] = [None] * 5
            
            # 存储结果（组号从0开始对应索引）
            results["variances"][key][group_id] = variance
            results["probabilities"][key][group_id] = prob
            results["entropies"][key][group_id] = entropy
    
    return results

class ExactAucCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, task_names):
        super(ExactAucCallback, self).__init__()
        self.validation_data = validation_data
        self.task_names = task_names
        self.auc_history = {name: [] for name in task_names}
    
    def on_epoch_end(self, epoch, logs=None):
        # 获取验证数据
        x_val, y_val_true = self.validation_data
        
        # 预测整个验证集
        y_val_pred = self.model.predict(x_val, verbose=0)
        
        # 如果是多输出模型，y_val_pred 是列表，y_val_true 也可能是列表
        if isinstance(y_val_pred, list):
            for i, task_name in enumerate(self.task_names):
                y_true = y_val_true[i] if isinstance(y_val_true, list) else y_val_true
                y_pred = y_val_pred[i]
                
                # 确保形状正确
                if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                    y_pred = np.squeeze(y_pred)
                
                # 计算精确 AUC
                try:
                    exact_auc = roc_auc_score(y_true, y_pred)
                    self.auc_history[task_name].append(exact_auc)
                    print(f"Epoch {epoch+1} - {task_name} Exact AUC: {exact_auc:.6f}")
                    
                    # 也可以添加到 logs 中以便在 History 对象中记录
                    logs[f'{task_name}_exact_auc'] = exact_auc
                except ValueError as e:
                    print(f"Could not calculate AUC for {task_name}: {e}")
        else:
            # 单输出模型
            try:
                exact_auc = roc_auc_score(y_val_true, y_val_pred)
                self.auc_history[self.task_names[0]].append(exact_auc)
                print(f"Epoch {epoch+1} - Exact AUC: {exact_auc:.6f}")
                logs['exact_auc'] = exact_auc
            except ValueError as e:
                print(f"Could not calculate AUC: {e}")

        
def sequential_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    按顺序将数据分割为训练集、验证集和测试集
    """
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    # 确保分割尺寸合理
    if train_size + val_size + test_size != total_samples:
        test_size += (train_size + val_size + test_size) - total_samples
    # 按顺序分割
    train = data.iloc[:train_size]
    val = data.iloc[train_size:train_size+val_size]
    test = data.iloc[train_size+val_size:]
    return train, val, test

def main(external_args=None):
    #set_gpu_memory_limit(limit_mb=25*1024)
    args = external_args
    myseed=args.myseed
    tasktype=args.tasktype
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    tf.random.set_seed(myseed) 
    random.seed(myseed)
    # 设置NumPy的随机种子
    np.random.seed(myseed)
    #参数处理
    

    #parser.add_argument('--result_file', type=str,default='')
    #args = parser.parse_args()

    # ============================ 数据加载与预处理 ============================
    print("Step 1: 加载并预处理数据...")
    # 1.1 设置数据集路径 (根据实际位置修改)
    DATA_DIR = "../../datasets/KuaiSAR_v2"
    USER_FILE = os.path.join(DATA_DIR, "user_features.csv")
    VIDEO_FILE1 = os.path.join(DATA_DIR, "item_features.csv")
    #VIDEO_FILE2 = os.path.join(DATA_DIR, "video_features_statistic_pure.csv")
    INTERACTION_FILE = os.path.join(DATA_DIR, "rec_inter.csv")

    # 加载用户数据
    user_df = pd.read_csv(USER_FILE)
    print(f"Loaded {len(user_df)} user records")
    
    # 加载视频数据
    video_df1 = pd.read_csv(VIDEO_FILE1)
    print(f"Loaded {len(video_df1)} video records1")
    '''
    video_df2 = pd.read_csv(VIDEO_FILE2)
    print(f"Loaded {len(video_df2)} video records2")'''
    
    # 加载交互数据
    interaction_df = pd.read_csv(INTERACTION_FILE,nrows=5000000)
    print(f"Loaded {len(interaction_df)} interaction records")
    
    # 合并数据
    data = interaction_df.merge(
        user_df, on="user_id", how="left"
    ).merge(
        video_df1, on="item_id", how="left"
    )

    data['click_target'] = data['click'].astype(int)
    data['like_target'] = data['like'].astype(int)
    data['forward_target'] = data['forward'].astype(int)
    data['follow_target'] = data['follow'].astype(int)
    data['playing_time'] = data['playing_time'].fillna(0.0)
    data['duration_ms'] = data['duration_ms'].fillna(0.0)
    data['longview_flag'] = np.where(
    data['duration_ms'] > 0,
    (data['playing_time'] / data['duration_ms']) > 0.8,
    False
    )
    data['long_view'] = data['longview_flag'].astype(int)
    
    sparse_features = [
    # 用户ID和属性
    "user_id","item_id","search_active_level","rec_active_level","onehot_feat1","onehot_feat2","author_id",
    "item_type","upload_type","first_level_category_id","second_level_category_id","third_level_category_id","fourth_level_category_id"
    ]   
    
    # 稠密特征 (Dense Features)
    dense_features = [
    # 用户属性（连续值或二值）
    "duration_ms","timestamp","upload_type"
    
    
    ]
    ###########################
    logdense_features = [
    "duration_ms","timestamp","upload_type"
    ]
    #############################
    data = data.sort_values('timestamp').reset_index(drop=True)
    train, val, test = sequential_split(data)  # 假设此函数正确拆分出训练/验证/测试集

    # 为避免SettingWithCopyWarning，显式创建副本
    train = train.copy()
    val = val.copy()
    test = test.copy()
    # 1. 处理user_active_degree
    active_degree_map = {'AD':3, 'NORMAL':4, 'LIVEGAME_PLAYBACK':2, 'UNKNOWN':1}
    for df in [train, val, test]:
        df['item_type'] = df['item_type'].map(active_degree_map).fillna(1)
    active_degree_map2 = {'ALBUM2020': 44, 'ALBUM2021': 2, 'ALBUM2022': 3, 'AiCutVideo': 4, 'Camera': 5, 'CommonAvatar': 6, 'CommonMilepost': 7, 'CommonRelationship': 8, 'CommonScene': 9, 'Copy': 10, 'FlashPhoto': 11, 'FollowShoot': 12, 'Import': 13, 'Karaoke': 14, 'Kmovie': 15, 'KwappShare': 16, 'LipsSync': 17, 'LiveClip': 18, 'LivePlayback': 19, 'LocalCollection': 20, 'LocalIntelligenceAlbum': 21, 'LongCamera': 22, 'LongImport': 23, 'LongOriginImport': 24, 'LongPicture': 25, 'OriginImport': 26, 'OriginPicture': 27, 'PhotoCopy': 28, 'PhotoOriginal': 29, 'PictureCopy': 30, 'PictureSet': 31, 'Recreation': 32, 'ResidentChange': 33, 'SUMMARY2021': 34, 'SameFrame': 35, 'ShareFromOtherApp': 36, 'ShootRecognition': 37, 'ShortCamera': 38, 'ShortImport': 39, 'ShortOriginImport': 40, 'Solitaire': 41, 'Status': 42, 'StoryMoodTemplate': 43, 'UNKNOWN': 1, 'Web': 45}
    for df in [train, val, test]:
        df['upload_type'] = df['upload_type'].map(active_degree_map2).fillna(1)
    # 2. 处理tag
    all_tags=[]
    for tags_str in train['caption']:
        if pd.notna(tags_str):  # 处理空值
            all_tags.extend(tags_str.split(','))
    tag_counts = pd.Series(all_tags).value_counts()
    tag_low_freq_threshold = 10  # tag低频阈值
    high_freq_tags = set(tag_counts[tag_counts >= tag_low_freq_threshold].index)
    vocab = sorted(set(high_freq_tags))
    vocab_size = len(vocab) + 1  # +1 用于未知标签
    tag_to_id = {tag: idx+1 for idx, tag in enumerate(vocab)}  # 从1开始编号
    tag_to_id['<UNK>'] = 0  # 未知标签映射到0
    def process_tag(df):
        return df['caption'].apply(
            lambda x: [tag_to_id.get(t, 0) for t in x.split(',')] if pd.notna(x) else []
        )
    train['caption'] = process_tag(train)
    val['caption'] = process_tag(val)
    test['caption'] = process_tag(test)
    # 4. 计算最大序列长度并填充序列
    max_len = 3#data['tag'].apply(len).max()
    train_tag_sequences = pad_sequences(
    train['caption'], 
    maxlen=max_len, 
    padding='post', 
    value=0
    )
    val_tag_sequences = pad_sequences(
    val['caption'], 
    maxlen=max_len, 
    padding='post', 
    value=0
    )
    test_tag_sequences = pad_sequences(
    test['caption'], 
    maxlen=max_len, 
    padding='post', 
    value=0
    )
    train['caption'] = train_tag_sequences.tolist()
    val['caption'] = val_tag_sequences.tolist()
    test['caption'] = test_tag_sequences.tolist()
    # 3. 处理upload_dt
    '''data['upload_dt'] = pd.to_datetime(data['upload_dt'])
    data['upload_year'] = data['upload_dt'].dt.year
    data['upload_month'] = data['upload_dt'].dt.month
    data['upload_day'] = data['upload_dt'].dt.day
    #sparse_features += ["upload_year", "upload_month", "upload_day"]'''
    # 从稀疏特征中移除时间原始列
    '''
    if "upload_dt" in sparse_features:
        sparse_features.remove("upload_dt")'''
    # 处理缺失值
    missing_columns = [col for col in dense_features if col not in data.columns]
    if missing_columns:
        print(f"警告: 以下列在数据框中缺失: {missing_columns}")
    for df in [train, val, test]:
        df[sparse_features] = df[sparse_features].fillna(-1)
        df[dense_features] = df[dense_features].fillna(0)
    feat_high_freq = {}
    # 稀疏特征标签编码
    for feat in sparse_features:
        '''train_classes = set(train[feat].unique())
        val[feat] = val[feat].apply(lambda x: x if x in train_classes else -1)
        test[feat] = test[feat].apply(lambda x: x if x in train_classes else -1)
        
        lbe = LabelEncoder()
        train[feat] = lbe.fit_transform(train[feat])
        val[feat] = lbe.transform(val[feat])
        test[feat] = lbe.transform(test[feat])'''
        #data[feat] = lbe.fit_transform(data[feat])
        # 4.1 统计训练集特征值频率，过滤低频
        value_counts = train[feat].value_counts()
        low_freq_threshold = 10  # 可根据特征调整（如user_id可用更高阈值）
        high_freq_values = set(value_counts[value_counts >= low_freq_threshold].index)
        
        # 4.2 保存高频列表
        high_freq_list = sorted(high_freq_values)
        feat_high_freq[feat] = high_freq_list
        
        # 4.3 构建映射（高频→1+，低频/未知→0）
        feat_map = {val: idx+1 for idx, val in enumerate(high_freq_list)}
        
        # 4.4 编码（训练集低频→0，参与UNK训练）
        train[feat] = train[feat].apply(lambda x: feat_map.get(x, 0))
        val[feat] = val[feat].apply(lambda x: feat_map.get(x, 0))
        test[feat] = test[feat].apply(lambda x: feat_map.get(x, 0))
    for feature in logdense_features:
        # 添加1是为了处理特征值为0的情况
        train[feature] = np.log(1+train[feature]) 
        val[feature] = np.log(1+val[feature])
        test[feature] = np.log(1+test[feature])
    # 稠密特征归一化
    
    mms = MinMaxScaler(feature_range=(0, 1))
    
    train[dense_features] = mms.fit_transform(train[dense_features])  
    val[dense_features] = mms.transform(val[dense_features])      
    test[dense_features] = mms.transform(test[dense_features])  
    #data[dense_features] = mms.fit_transform(data[dense_features])  
    #"""为DeepCTR模型创建特征列"""
    sparse_feat_list = [
        SparseFeat(
            feat, 
            vocabulary_size=len(feat_high_freq[feat]) + 1,  # +1 用于未知类别
            embedding_dim=args.embedding_dimension
        ) for feat in sparse_features
    ]
    temp=[VarLenSparseFeat(SparseFeat('caption', vocabulary_size=vocab_size, embedding_dim=args.embedding_dimension),combiner='mean',maxlen=max_len)]
    
    dense_feat_list = [DenseFeat(feat, 1) for feat in dense_features]

    '''
    # 3.generate input data for model
    data['tag']=tag_sequences.tolist()
    #print("counts:")
    #print(data['counts'])
    #print("play_cnt:")
    #print(data['play_cnt'])
    #train, test = train_test_split(data, test_size=0.2, random_state=1024)
    data = data.sort_values('time_ms').reset_index(drop=True)'''

    '''ts=data['time_ms']
    for i in range(len(ts)-1):
        if ts[i]>ts[i+1]:
            print(i)'''
    '''train, val, test = sequential_split(data)
    train = train.copy()
    val = val.copy()
    test = test.copy()'''
    
    feature_columns=sparse_feat_list+dense_feat_list+temp
    feature_names = get_feature_names(feature_columns)
    train_model_input = {name: train[name] for name in feature_names}
    train_model_input['caption']=np.array(train['caption'].tolist())

    val_model_input = {name: val[name] for name in feature_names}
    val_model_input['caption']= np.array(val['caption'].tolist())
    test_model_input = {name: test[name] for name in feature_names}
    test_model_input['caption']=np.array(test['caption'].tolist())

    # 定义任务配置
    task_type = {
        "click_target": "binary",
        "long_view": "binary",
        "like_target": "binary",
        "follow_target": "binary",
        "forward_target": "binary"  # === 最后一个是主任务 ===
    }
    task_names = list(task_type.keys())
    task_types = list(task_type.values())

    main_task = task_names[-1]

    datasets = {
    'train': train,
    'valid': val, 
    'test': test
    }

    # 目标列列表
    target_columns = ['click_target', 'long_view', 'like_target', 'forward_target', 'follow_target']
    results = pd.DataFrame(index=target_columns, columns=datasets.keys())

    # 计算每个数据集中每个目标的正样本数量
    for dataset_name, dataset in datasets.items():
        for target in target_columns:
            if target in dataset.columns:
                # 计算正样本数量（假设正样本标记为1）
                positive_count = (dataset[target] == 1).sum()
                results.loc[target, dataset_name] = positive_count
            else:
                results.loc[target, dataset_name] = 'N/A'

    # 打印结果
    print("各数据集中正样本数量统计:")
    print(results)




    # 初始化PLE模型
    model = mymodel4(
        feature_columns,
        bottom_dnn_hidden_units=(512,), tower_dnn_hidden_units=(128,64,32),
        l2_reg_embedding=args.l2_regularization,
        l2_reg_dnn=args.l2_regularization, seed=myseed, dnn_dropout=0.1, dnn_activation='relu', 
        dnn_use_bn=True,
        task_types=task_types,
        task_names=task_names,
        tasktype=tasktype
    )
    def create_model(feature_columns, task_types, task_names):
        return mymodel2(
        feature_columns,
        bottom_dnn_hidden_units=(512,), tower_dnn_hidden_units=(128,64,32),
        l2_reg_embedding=args.l2_regularization,
        l2_reg_dnn=args.l2_regularization, seed=myseed, dnn_dropout=0.1, dnn_activation='relu', 
        dnn_use_bn=True,
        task_types=task_types,
        task_names=task_names,
        tasktype=tasktype
    )
    
    model.compile(
        optimizer='adam',
        loss={"click_target": "binary_crossentropy", 
            "long_view": "binary_crossentropy",
            "like_target": "binary_crossentropy", 
            "forward_target": "binary_crossentropy", 
            "follow_target": "binary_crossentropy"},
        metrics={"click_target": ["AUC","binary_crossentropy",], 
            "long_view": ["AUC","binary_crossentropy",],
            "like_target": ["AUC","binary_crossentropy",], 
            "forward_target": ["AUC","binary_crossentropy",], 
            "follow_target": ["AUC","binary_crossentropy",]},
        loss_weights={"click_target": 1, 
            "long_view": 1,
            "like_target": 1, 
            "forward_target": 2, 
            "follow_target": 1}
    )
    model.optimizer.learning_rate = args.learning_rate
    class LateEarlyStopping(tf.keras.callbacks.EarlyStopping):
        def __init__(self, start_epoch=50, **kwargs):
            super().__init__(**kwargs)
            self.start_epoch = start_epoch  # 从第start_epoch轮开始生效

        def on_epoch_end(self, epoch, logs=None):
            # 前start_epoch轮：强制重置计数器，不触发早停
            if epoch < self.start_epoch:
                self.counter = 0  # 重置早停计数
                return
            # 第start_epoch轮后：执行原生早停逻辑
            super().on_epoch_end(epoch, logs)
    early_stop = EarlyStopping(
    #start_epoch=20,
    monitor=f'val_{main_task}_loss',    # 监控验证集损失
    patience=5,           # 允许连续3轮不改善
    verbose=1,            # 打印日志
    mode='min',           # 监控指标越小越好
    restore_best_weights=True  # 恢复最佳模型权重
    )      

    # 可选：添加 ModelCheckpoint 保存最佳模型
    # 创建唯一文件名标识符
    def create_unique_filename(base_name):
    # 使用时间戳 + UUID + 进程ID确保唯一性
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        pid = os.getpid()
        return f"trash/{base_name}_{timestamp}_{pid}_{unique_id}.h5"
    weights_filename = create_unique_filename('best_model_weights')
    checkpoint = ModelCheckpoint(
    weights_filename,
    monitor=f'val_{main_task}_loss',
    save_best_only=True,
    save_weights_only=True,  # 只保存权重
    mode='min'
    )
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    trian_values = [train[v].values for v in task_names]
    val_values = [val[v].values for v in task_names]

    # exact_auc_callback = ExactAucCallback(
    # validation_data=(val_model_input, [val['click_target'].values, val['long_view'].values, val['like_target'].values,val['forward_target'].values,val['follow_target'].values]),
    # task_names=task_names
    # )
    exact_auc_callback = ExactAucCallback(
        validation_data=(val_model_input, val_values),
        task_names=task_names
        )
    # 训练模型

    history = model.fit(
        train_model_input,
        trian_values,
        validation_data=(
        val_model_input,  # 验证集输入
        val_values),  # 验证集标签
        batch_size=args.batch_size,
        epochs=250,
        verbose=1,
        callbacks=[ checkpoint,tensorboard_callback,early_stop]#
    )   
    # history = model.fit(
    #     train_model_input,
    #     [train['click_target'].values, train['long_view'].values, train['like_target'].values,train['forward_target'].values,train['follow_target'].values],
    #     validation_data=(
    #     val_model_input,  # 验证集输入
    #     [val['click_target'].values, val['long_view'].values, val['like_target'].values,val['forward_target'].values,val['follow_target'].values]),  # 验证集标签
    #     batch_size=args.batch_size,
    #     epochs=250,
    #     verbose=1,
    #     callbacks=[ checkpoint,tensorboard_callback,early_stop]#
    # )   

    pred_ans = model.predict(test_model_input, batch_size=args.batch_size)
    print("test click_target AUC", round(roc_auc_score(test['click_target'], pred_ans[0]), 4))
    print("test click_target logloss", round(log_loss(test['click_target'], pred_ans[0]), 4))
    print("test long_view AUC", round(roc_auc_score(test['long_view'], pred_ans[1]), 4))
    print("test long_view logloss", round(log_loss(test['long_view'], pred_ans[1]), 4))
    print("test like_target AUC", round(roc_auc_score(test['like_target'], pred_ans[2]), 4))
    print("test like_target logloss", round(log_loss(test['like_target'], pred_ans[2]), 4))
    print("test forward_target AUC", round(roc_auc_score(test['forward_target'], pred_ans[3]), 4))
    print("test forward_target logloss", round(log_loss(test['forward_target'], pred_ans[3]), 4))
    print("test follow_target AUC", round(roc_auc_score(test['follow_target'], pred_ans[4]), 4))
    print("test follow_target logloss", round(log_loss(test['follow_target'], pred_ans[4]), 4))
    


    log_result = {
        "click_target_AUC":round(roc_auc_score(test['click_target'], pred_ans[0]), 4),
        "click_target_logloss" : round(log_loss(test['click_target'], pred_ans[0]), 4),
        "long_view_AUC": round(roc_auc_score(test['long_view'], pred_ans[1]), 4),
        "long_view_logloss": round(log_loss(test['long_view'], pred_ans[1]), 4),
        "like_target_AUC":round(roc_auc_score(test['like_target'], pred_ans[2]), 4),
        "like_target_logloss" : round(log_loss(test['like_target'], pred_ans[2]), 4),
        "forward_target_AUC":round(roc_auc_score(test['forward_target'], pred_ans[3]), 4),
        "forward_target_logloss" : round(log_loss(test['forward_target'], pred_ans[3]), 4),
        "follow_target_AUC":round(roc_auc_score(test['follow_target'], pred_ans[4]), 4),
        "follow_target_logloss" : round(log_loss(test['follow_target'], pred_ans[4]), 4)
    }
    if args.result_file:
        
        with open(args.result_file,'a',newline='') as f:
            #fcntl.flock(f, fcntl.LOCK_EX)  # 加锁
            writer = csv.writer(f)
            
            
            
            # 结果值
            row=[
            args.myseed,
            log_result.get('click_target_AUC', ''),
            log_result.get('click_target_logloss', ''),
            log_result.get('long_view_AUC', ''),
            log_result.get('long_view_logloss', ''),
            log_result.get('like_target_AUC', ''),
            log_result.get('like_target_logloss', ''),
            log_result.get('forward_target_AUC', ''),
            log_result.get('forward_target_logloss', ''),
            log_result.get('follow_target_AUC', ''),
            log_result.get('follow_target_logloss', ''),
            len(history.epoch)
            ]
            
            writer.writerow(row)
            #fcntl.flock(f, fcntl.LOCK_UN)  # 解锁


    
    
    true_labels = {name: test[name].values for name in task_names}

    # 存储所有结果的字典
    all_results = {
    "variances": {},
    "probabilities": {},
    "entropies": {}
    }

    # 遍历每个任务（0-4）
    for i in range(5):
        print(f"正在处理任务{i}：{task_names[i]}")
        # 获取当前任务的预估值和边界
        current_pred = pred_ans[i]
        task_results = calculate_group_metrics(
        task_index=i,
        pred_values=current_pred,
        true_labels=true_labels,
        task_names=task_names
        )
    
        # 合并到总结果
        for metric in ["variances", "probabilities", "entropies"]:
            for key, value in task_results[metric].items():
                all_results[metric][key] = value
        

def run_experiment(gpu_id, result_file,k,tasktype):
    """在指定GPU上运行实验的函数"""
    # 设置当前进程使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    seed = 1024+gpu_id+k*8
    run_params = {
        "learning_rate":0.00001,
        "embedding_dimension": 20,
        "batch_size":512,
        "l2_regularization": 0.01,
        'myseed': seed,
        'result_file': result_file,
        'tasktype':tasktype
    }
    
    #args = parser.parse_args([])  # 传入空列表因为我们直接设置参数值
    args = argparse.Namespace(**run_params)
    
    # 调用主函数并传入参数
    main(external_args=args)


def calculate_statistics(result_file):
    """计算并打印结果的统计信息"""
    # 读取所有结果
    results = []
    with open(result_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # 跳过标题行（如果有）
            if not row[0].replace('.', '').isdigit():
                continue
            results.append([float(x) if x != '' else 0.0 for x in row])
    
    if not results:
        print("没有找到结果数据")
        return
    
    # 转置矩阵以便按列计算
    results = np.array(results).T
    
    # 定义指标名称（根据你的输出顺序）
    metric_names = [
        "seed",
        "Click Target AUC",
        "Click Target LogLoss",
        "Long View AUC",
        "Long View LogLoss",
        "Like Target AUC",
        "Like Target LogLoss",
        "Commentforward Target AUC",
        "Commentforward Target LogLoss",
        "Follow Target AUC",
        "Follow Target LogLoss",
    ]
    
    print("\n" + "="*50)
    print("实验结果统计摘要")
    print("="*50)
    
    # 计算并打印每个指标的统计信息
    for i, (name, values) in enumerate(zip(metric_names, results)):
        # 前4个是参数，不需要计算统计量
        if i < 1:
            continue
            
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        print(f"{name:<20}: 均值 = {mean_val:.6f}, 标准差 = {std_val:.6f}")
    
    print("="*50)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 2. 定义需要接收的参数
    # 第一个参数：CSV文件名（类型为str）
    parser.add_argument('csv_file', type=str)

    # 3. 解析参数
    args = parser.parse_args()
    result_file = args.csv_file
    tasktype=10
    
    # 3. 清空或创建结果文件（只保留标题）
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'seed',
            'click_auc', 'click_logloss',
            'long_view_auc', 'long_view_logloss',
            'like_auc', 'like_logloss',
            'commentforward_auc', 'commentforward_logloss',
            'follow_auc', 'follow_logloss',
        ])
        
    processes = []
    gpu_ids_list = [0,1]
    for i in gpu_ids_list:
        #if i!=4 and i!=5 and i!=7:
        #    continue
        p = Process(target=run_experiment, args=(i, result_file,0,tasktype))
        p.start()
        processes.append(p)
        print(f"启动进程 {i} 在 GPU {i}")
        time.sleep(5)
    for p in processes:
        p.join(timeout=10800)  # 1小时超时
        if p.is_alive():
            print(f"进程 {p.pid} 超时，终止")
            p.kill()
