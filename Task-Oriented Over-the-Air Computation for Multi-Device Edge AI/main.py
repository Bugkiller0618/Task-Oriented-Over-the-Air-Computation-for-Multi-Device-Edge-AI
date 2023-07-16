import numpy as np
from numpy.random import default_rng
import cvxpy as cp
from sklearn import datasets, svm, metrics
import pickle


#load mean, variance, test data, and label
mean_class = np.load('./save_model/save_mean_variance/mean_class_12dim.npy', allow_pickle=True)
var_class = np.load('./save_model/save_mean_variance/var_class_12dim.npy', allow_pickle=True)
data_test_pca_normed = np.load('./save_model/save_mean_variance/data_test_pca_normed_12dim.npy', allow_pickle=True)
label_test = np.load('./save_model/save_mean_variance/label_test_12dim.npy', allow_pickle=True)

# parameters
num_antenna = 8  # number of antenna, N
PCA_dim = 12  # PCA dimension, M
num_device = 3  # number of devices, K
num_class = 4  # number of classes, L

var_dist_scale = 0.4
var_comm_noise = 1  # communication noise, sigma_{0}^{2}
rng = default_rng(2022)
var_dist = rng.uniform(0, var_dist_scale, (num_device, PCA_dim))  # variance of distortion, delta_{k,m}

power_mdB = 12
power_tx = 10 ** ( (power_mdB-30) / 10 )  # transmit power, P_{k}
#power_tx = 0.1  # transmit power, P_{k}
bandwidth = 1.5 * 10**6

radius = 500  # BS in the center of circle of radius = 500 m
radius_inner = 450  # Distance between device circle and the server
chl_shadow_std_db = 8  # shadow fading standard deviation = 8 dB
user_dist = (radius - radius_inner) * rng.random((num_device, 1)) + radius_inner
user_pl_db = 128.1 + 37.6 * np.log10(user_dist / 1e3)  # path loss in dB
user_pl_db = user_pl_db - chl_shadow_std_db
user_pl = 10 ** (-user_pl_db / 10)
rayli_fading_real = rng.normal(0, 1, (num_device, num_antenna))  # rayleigh fading ~ CN(0,1)
rayli_fading_img = rng.normal(0, 1, (num_device, num_antenna))
rayli_fading_gain = rayli_fading_real ** 2 + rayli_fading_img ** 2
noise_power = 10**(-17.4) * bandwidth  # from Paper
channel_gain = user_pl * np.ones((1, num_antenna)) * np.sqrt(rayli_fading_gain) / noise_power

#two dimension's PCA
def SCA_for_two_PCA(num_antenna,power_tx,channel_gain,num_class,idx,num_device):
    # initialization
    f_vec_init = np.ones((num_antenna, 1))  # beamforming init, f
    c_zf_init = np.sqrt(2 * power_tx * channel_gain ** 2 @ (f_vec_init ** 2))
    alpha_init = np.zeros(( int (num_class * (num_class-1) / 2), 2))
    for idx_class_a in range(num_class):
        for idx_class_b in range(idx_class_a):
            for idx_m in range(2):
                alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m] = ((mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) ** 2) / ((np.sum(var_dist[:, 2 * idx + idx_m].reshape(1,num_device) @ (c_zf_init ** 2)) + var_comm_noise * np.sum(f_vec_init ** 2)) / ((np.sum(c_zf_init)) ** 2) + var_class[2 * idx + idx_m])

    disc_init = 2*alpha_init.sum() / num_class / (num_class-1)

    last_value = disc_init
    diff = last_value
    count = 1

    while(diff>1e-2 and count<100):
        # optimization
        f_vec = cp.Variable((num_antenna, 1))
        c_zf = cp.Variable((num_device, 1))
        alpha = {}
        for idx_m in range(2):
            alpha[idx_m] = cp.Variable((int(num_class * (num_class-1) / 2), 1))

        constrains = []
        for idx_device in range(num_device):
            channel_gain_device = channel_gain[idx_device, :].reshape((-1, 1))
            constrains += [c_zf[idx_device] ** 2 - 2 * power_tx * channel_gain_device.T @ f_vec_init @ f_vec_init.T @ channel_gain_device - 4 * power_tx * channel_gain_device.T @ f_vec_init @ channel_gain_device.T @ (f_vec - f_vec_init) <= 0]

        for idx_class_a in range(num_class):
            for idx_class_b in range(idx_class_a):
                for idx_m in range(2):
                    constrains += [(cp.sum_squares(cp.multiply(c_zf, np.sqrt(var_dist[:, 2 * idx + idx_m].reshape(-1, 1)))) + var_comm_noise * cp.sum_squares(f_vec)) + (cp.sum(c_zf)) ** 2 * var_class[2 * idx + idx_m] - np.sum(c_zf_init) ** 2 * ((mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) ** 2 / alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m]) - 2 * (mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) ** 2 * np.sum(c_zf_init) / alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m] * cp.sum(c_zf - c_zf_init) + (np.sum(c_zf_init) * (mean_class[idx_class_a, 2 * idx + idx_m] - mean_class[idx_class_b, 2 * idx + idx_m]) / alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m]) ** 2 * (alpha[idx_m][int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b] - alpha_init[int(idx_class_a * (idx_class_a-1) / 2) + idx_class_b, idx_m]) <= 0]

        for idx_device in range(num_device):
            constrains += [c_zf[idx_device] >= 0]

        for idx_antenna in range(num_antenna):
            constrains += [f_vec[idx_antenna] >= 0]

        for idx_class_a in range(num_class):
            for idx_class_b in range(idx_class_a):
                for idx_m in range(2):
                    constrains += [alpha[idx_m][int(idx_class_a * (idx_class_a - 1) / 2) + idx_class_b] >= 0]

        objective = cp.Maximize(2 * cp.sum(alpha[0] + alpha[1]) / num_class / (num_class-1))
        prob = cp.Problem(objective, constrains)
        prob.solve(solver=cp.SCS, verbose=False)

        stepsize = 0.1

        f_vec_init = f_vec.value * stepsize + f_vec_init * (1-stepsize)
        c_zf_init = c_zf.value * stepsize + c_zf_init * (1-stepsize)
        temp1 = alpha[0].value * stepsize
        alpha_init[:, 0] = temp1.reshape(int(num_class * (num_class-1) / 2)) + alpha_init[:, 0] * (1-stepsize)
        temp2 = alpha[1].value * stepsize
        alpha_init[:, 1] = temp2.reshape(int(num_class * (num_class-1) / 2)) + alpha_init[:, 1] * (1-stepsize)

        diff = abs(prob.value - last_value)
        last_value = prob.value
        print("The discriminant gain after {}-th interation is: {}".format(count, last_value))
        count += 1

    #print("The discriminant gain after optimization is: {}".format(prob.value))
    return c_zf_init,f_vec_init,last_value

def add_noise_to_normed_pca(data_test_pca_normed, c_zf_init, f_vec_init):
    var_dist = rng.uniform(0, var_dist_scale, (num_device, 2))
    data_test_pca_add_noise = (np.sum(c_zf_init) * data_test_pca_normed + c_zf_init.T @ var_dist + np.sum(f_vec_init * var_comm_noise)) / np.sum(c_zf_init)
    return data_test_pca_add_noise

def model_inference(data, label, model):
    predicted = model.predict(data)
    accuracy = metrics.accuracy_score(label, predicted)
    return accuracy


#####the below is computating the accuracy with the change of power#####
power_mdB = np.linspace(0,20,11)
power_list = 10 ** ( (power_mdB-30) / 10 )
np.save('./save_model/save_results/power_list.npy', power_mdB)
svm_accuracy_init_list = np.zeros((len(power_list), 1))
mlp_accuracy_init_list = np.zeros((len(power_list), 1))
discriminant_gain_list = np.zeros((len(power_list), 1))

for i in range(len(power_list)):
    power_tx = power_list[i]
    data_test_pca_add_noise = np.ones((np.size(data_test_pca_normed,0),PCA_dim))
    discriminant_gain_init = []

    for idx in range(0,int(PCA_dim/2)):
        c_zf, f_vec, discri = SCA_for_two_PCA(num_antenna,power_tx,channel_gain,num_class,idx,num_device)
        discriminant_gain_init.append(discri)
        data_test_pca_add_noise[:,idx*2:idx*2+2] = add_noise_to_normed_pca(data_test_pca_normed[:,idx*2:idx*2+2], c_zf, f_vec)

    discriminant_gain_list[i] = np.sum(discriminant_gain_init)

    svm_model_file = './save_model/svm_model_{}dimension.pkl'.format(PCA_dim)
    mlp_model_file = './save_model/mlp_model_{}dimension.pkl'.format(PCA_dim)

    with open(svm_model_file, 'rb') as file:
        svm_model = pickle.load(file)
    with open(mlp_model_file, 'rb') as file:
        mlp_model = pickle.load(file)

    svm_accuracy_init_list[i] = model_inference(data_test_pca_add_noise, label_test, svm_model)
    mlp_accuracy_init_list[i] = model_inference(data_test_pca_add_noise, label_test, mlp_model)


np.save('./save_model/save_results/svm_accuracy_with_power.npy', svm_accuracy_init_list)
np.save('./save_model/save_results/mlp_accuracy_with_power.npy', mlp_accuracy_init_list)
np.save('./save_model/save_results/discriminant_gain_with_power.npy', discriminant_gain_list)