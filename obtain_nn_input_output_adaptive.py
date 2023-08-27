import numpy as np
def obtain_nn_input_output_adaptive(inputdata_r,inputdata_i,outputdata_r,outputdata_i,M,HYPER):
    temp1=inputdata_r
    temp2=inputdata_i
    for i in range(M):

        temp1_=np.vstack([np.zeros((i+1,1)),inputdata_r[0:-1*i-1]])
        temp1=np.hstack([temp1,temp1_])
        temp2_=np.vstack([np.zeros((i+1,1)),inputdata_i[0:-1*i-1]])
        temp2=np.hstack([temp2,temp2_])

    temp3=np.hstack([outputdata_r,outputdata_i])
    temp=np.hstack([temp1,temp2])

    for j in range(np.size(HYPER)):
        temp=np.hstack([temp,np.ones((np.size(temp,0),1))*HYPER[j]])


    return temp, temp3






