from pytorch.tools import data_int # 自己的工具包
import pandas as pd
csv_path = 'C:\\Users\\Icsm\\Desktop\\kddcup99毕设相关\\数据\\kddcup.data_10_percent_corrected.csv'
# 先得到新的数据集， 并写入到csv文件中
data = data_int(csv_path)

# 选取新的特征值
# TCP连接基本特征:
tcp_connect_lists = ['duration', # 连接持续时间
                     'protocol_type', # 协议类型
                     'service', # 目标主机网络服务类型
                     'flag', # 连接正常或错误状态
                     'src_bytes', # 源到目标主机字节数
                     'dst_bytes', # 目标到源字节数
                     'land', # 若连接及送达都为同一个主机则为1
                     'wrong_fragment', # 错误的分段数量
                     'urgent'] # 加急包个数
# TCP连接内容特征
# 基于时间的网络流量统计特征
# 基于主机的网络流量统计特征

# data[tcp_connect_lists].to_csv('./tcp_connect.csv')

# 全部数据集用于分类

columns_list = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent',
                'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_hot_login',
                'is_guest_login',
                'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
                'label']
data[columns_list].to_csv('./kddcup99_10%.csv')


