import pandas as pd
import numpy as np



def data_int(csv_path):
    '''
    :param csv_path:
    :return: 返回string类型数据变为数值类型
    '''
    data = pd.read_csv(csv_path,header=None,engine='python')
    '''
    read_csv()参数问题:
        1、header=None 认为csv文件没有索引
        2、index_col=0 则指定csv文件第一行数据为索引
        3、engine='python'这是路径有中文时，需要添加
    '''
    columns_list=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
               'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_hot_login','is_guest_login',
               'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
               'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
                'label']
    data.columns = columns_list #给数据添加列名
    service_list = ['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u',
                     'echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames',
                     'http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap',
                     'link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp',
                     'ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell',
                     'smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i',
                     'uucp','uucp_path','vmnet','whois','X11','Z39_50']
    service_dict = dict([[service,index] for index,service in enumerate(service_list)])#转换为字典格式，用于下面的字符格式转换为数值类型(字符类型无法用于算法计算),enumerate()返回索引及数值
    flag_list = ['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    flag_dict = dict([[flag,index] for index,flag in enumerate(flag_list)])
    label_list=['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
         'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
         'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
         'spy.', 'rootkit.']
    label_dict = dict([[label,index] for index,label in enumerate(label_list)])
    #将dataframe数据字符类型转换为数值类型
    data['protocol_type'] = data['protocol_type'].map({'tcp':0,'udp':1,'icmp':2})
    data['service'] = data['service'].map(service_dict)
    data['flag'] = data['flag'].map(flag_dict)
    data['label'] = data['label'].map(label_dict)
    return data


def split_test_data(csv_path, to_csv_path, start, end):
    '''
    :param csv_path: 源数据csv文件
    :param to_csv_path: 存储数据csv文件
    :param start: 截取开始
    :param end: 截取结束
    :return:
    '''
    data = data_int(csv_path)
    features_list = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
                   'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_hot_login','is_guest_login',
                   'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
                   'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']
    data = data[features_list]
    test_x = data[start:end]
    test_x.to_csv(to_csv_path)
