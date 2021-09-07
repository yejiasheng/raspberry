import os
import threading
def a():
    os.system("python /home/YJS/111/init_raspberry.py &")
def b():
    os.system("python /home/YJS/111/mergeyjs.py &")
# threads=[]
# threads.append(threading.Thread(target=a))
# threads.append(threading.Thread(target=b))
# print(threads)

if __name__ == '__main__':
    # for t in threads:
    #     t.start()
    # t.join()
    a()
    b()