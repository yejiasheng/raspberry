import paramiko
#############################配置信息#####################################
# 登陆参数设置
hostname = "192.168.43.250"  
username = "pi"
password = "raspberry"
########################################################################

def ssh_client_con():
    """创建ssh连接，并执行shell指令"""
    # 1 创建ssh_client实例
    ssh_client = paramiko.SSHClient()
    # 自动处理第一次连接的yes或者no的问题
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    # 2 连接服务器
    ssh_client.connect(
        hostname=hostname,
        port=22,
        username=username,
        password=password
    )
    # 3 执行shell命令
    # 构造shell指令ç
    # shell_command = "python /home/pi/CLBROBOT/srqtest2.py"
    shell_command = "python /home/pi/CLBROBOT/yjstest.py"
    # 执行shell指令
    stdin, stdout, stderr = ssh_client.exec_command(shell_command)
    # 输出返回信息
    stdout_info = stdout.read().decode('utf8')
    print(stdout_info)
    # 输出返回的错误信息
    stderr_info = stderr.read().decode('utf8')
    print(stderr_info)
ssh_client_con()