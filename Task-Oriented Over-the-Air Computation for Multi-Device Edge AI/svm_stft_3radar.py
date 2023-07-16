#  based on svm_stft.py, data from 3 radars

import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from data_set import *
import scipy.io as sio
from sklearn.decomposition import PCA
from numpy.random import default_rng
import pickle


def raw_resample_real_whitening(mat):
    mat_data = []
    sample = np.zeros(1520)
    for ii in range(len(mat[:, ...])):
        # resample: 400*38 --> 40 * 38 --> 1520*1
        sample = np.abs(mat[ii, 0:400:10, :].real).reshape(-1,) + np.abs(mat[ii, 0:400:10, :].imag).reshape(-1,)
        # sample[1:3040:2] = mat[ii, 0:400:10, :].imag.reshape(-1,)
        mean = np.mean(sample)
        std = np.max([np.std(sample), 1.0 / np.sqrt(1520)])
        mat_data.append((sample - mean) / std)
    return mat_data


def data_from_mat(mat_dir):
    data_train = []
    data_test = []
    label_train = []
    label_test = []
    dir_list = sorted(os.listdir(mat_dir))
    dir_list.remove(dir_list[4])
    for mat_name in dir_list:
        mat_fname = os.path.join(mat_dir, mat_name)
        mat_contents = sio.loadmat(mat_fname)
        data_train = data_train + raw_resample_real_whitening(mat_contents['mat'][:800, ...])
        data_test = data_test + raw_resample_real_whitening(mat_contents['mat'][800:, ...])
        label_train = label_train + [mat_name[-5]] * 800
        label_test = label_test + [mat_name[-5]] * 200
    return data_train, data_test, label_train, label_test


def gen_pca_model(data, pca_model, PCA_dim):
    if os.path.exists(pca_model) is False:
        # PCA
        pca_spec_raw = PCA(n_components=PCA_dim)
        pca_spec_raw.fit(data)
        with open(pca_model, 'wb') as file:
            pickle.dump(pca_spec_raw, file)
    else:
        with open(pca_model, 'rb') as file:
            pca_spec_raw = pickle.load(file)
    return pca_spec_raw


def add_noise_to_normed_pca(data_test_pca_normed, power_tx, power_quantization):
    # Constant parameters
    variance_receiver = 1  # \sigma_{r}^{2}
    variance_clutter = np.array([1, 0.5])  # \sigma_{c,k}^{2}
    variance_quantization = np.ones(1)  # \delta_{k}^{2}
    num_radar = 2

    # receiver noise
    rng = default_rng(2021)
    noise_receiver = np.zeros_like(data_test_pca_normed)
    for idx_radar in range(num_radar):
        noise_receiver[:, idx_radar * PCA_dim:(idx_radar + 1) * PCA_dim] = rng.normal(0, variance_receiver / power_tx[idx_radar], (num_test_per_class * num_class, PCA_dim))

    # clutter noise
    noise_clutter = np.zeros_like(data_test_pca_normed)
    for idx_radar in range(num_radar):
        noise_clutter[:, idx_radar * PCA_dim:(idx_radar + 1) * PCA_dim] = rng.normal(0, variance_clutter[idx_radar], (num_test_per_class * num_class, PCA_dim))

    # quantization noise
    noise_quantization = np.zeros_like(data_test_pca_normed)
    for idx_radar in range(num_radar):
        noise_quantization[:, idx_radar * PCA_dim:(idx_radar + 1) * PCA_dim] = rng.normal(0, variance_quantization[idx_radar] / power_quantization[idx_radar], (num_test_per_class * num_class, PCA_dim))

    noise = noise_receiver + noise_clutter + noise_quantization
    # noise_level = 0
    # noise_base = class_mean_var[1, :]
    # noise_cov = noise_level*np.diag(noise_base)
    # noise = rng.multivariate_normal(np.zeros(PCA_dim*3), noise_cov, (1000,))

    data_test_pca_add_noise = data_test_pca_normed + noise
    return data_test_pca_add_noise


def svm_inference(model, data, label):
    predicted = model.predict(data)
    accuracy = metrics.accuracy_score(label, predicted)
    return accuracy


def mlp_inference(model, data, label):
    predicted = model.predict(data)
    accuracy = metrics.accuracy_score(label, predicted)
    return accuracy



# dataset from mat files
data_dir_1 = '/home/liu/PycharmProjects/ISAC-train/data/spect/THREE_RADAR_STFT_MAT/radar_1'
data_dir_2 = '/home/liu/PycharmProjects/ISAC-train/data/spect/THREE_RADAR_STFT_MAT/radar_2'
data_dir_3 = '/home/liu/PycharmProjects/ISAC-train/data/spect/THREE_RADAR_STFT_MAT/radar_3'
data_train_1, data_test_1, label_train_1, label_test_1 = data_from_mat(data_dir_1)
data_train_2, data_test_2, label_train_2, label_test_2 = data_from_mat(data_dir_2)
data_train_3, data_test_3, label_train_3, label_test_3 = data_from_mat(data_dir_3)


# save pca model
PCA_dim = 10   #换dim要删除下面的三个文件才行
pca_model_1 = './save_model/pca_model_1_{}dim.pkl'.format(PCA_dim)
pca_model_2 = './save_model/pca_model_2_{}dim.pkl'.format(PCA_dim)
pca_model_3 = './save_model/pca_model_3_{}dim.pkl'.format(PCA_dim)
pca_spec_raw_1 = gen_pca_model(data_train_1, pca_model_1, PCA_dim)
pca_spec_raw_2 = gen_pca_model(data_train_2, pca_model_2, PCA_dim)
pca_spec_raw_3 = gen_pca_model(data_train_3, pca_model_3, PCA_dim)


# transform data into PCA axis
data_train_pca_1 = pca_spec_raw_1.transform(data_train_1)
data_test_pca_1 = pca_spec_raw_1.transform(data_test_1)

data_train_pca_2 = pca_spec_raw_2.transform(data_train_2)
data_test_pca_2 = pca_spec_raw_2.transform(data_test_2)

data_train_pca_3 = pca_spec_raw_3.transform(data_train_3)
data_test_pca_3 = pca_spec_raw_3.transform(data_test_3)

num_radar = 1
# if num_radar == 3:
#     data_train_pca = np.hstack((data_train_pca_1, data_train_pca_2, data_train_pca_3))
#     data_test_pca = np.hstack((data_test_pca_1, data_test_pca_2, data_test_pca_3))
# elif num_radar == 2:
#     data_train_pca = np.hstack((data_train_pca_1, data_train_pca_2))
#     data_test_pca = np.hstack((data_test_pca_1, data_test_pca_2))
# elif num_radar == 1:
#     data_train_pca = data_train_pca_1
#     data_test_pca = data_test_pca_1
# else:
#     print('The number of radar does not exit')

num_class = 4
num_train_per_class = 800
num_test_per_class = 200
data_train_pca = np.vstack((data_train_pca_1[0:num_train_per_class,:],data_train_pca_2[0:num_train_per_class,:]))
label_train = label_train_1[0:num_train_per_class] * 2
data_test_pca = np.vstack((data_test_pca_1[0:num_test_per_class,:],data_test_pca_2[0:num_test_per_class,:]))
label_test = label_test_1[0:num_test_per_class] * 2

for i in range(1,num_class):
    data_train_pca = np.vstack((data_train_pca,data_train_pca_1[i*num_train_per_class:(i+1)*num_train_per_class,:],data_train_pca_2[i*num_train_per_class:(i+1)*num_train_per_class,:]))

    label_train = label_train + label_train_1[i*num_train_per_class:(i+1)*num_train_per_class] * 2

    data_test_pca = np.vstack((data_test_pca,data_test_pca_1[i*num_test_per_class:(i+1)*num_test_per_class,:],data_test_pca_2[i*num_test_per_class:(i+1)*num_test_per_class,:]))

    label_test = label_test + label_test_1[i*num_test_per_class:(i+1)*num_test_per_class] * 2


num_class = 4
mean_class = np.zeros((num_class, PCA_dim*num_radar))  # mean of ground-true feature
var_class = np.zeros(PCA_dim*num_radar)  # variance of ground-true feature
num_train_per_class = 800*2
num_test_per_class = 200*2
for idx_class in range(num_class):
    class_data_train = data_train_pca[num_train_per_class * idx_class:num_train_per_class * (idx_class+1), :]
    class_data_test = data_test_pca[num_test_per_class * idx_class:num_test_per_class * (idx_class + 1), :]
    class_data = np.vstack((class_data_train, class_data_test))
    mean_class[idx_class, :] = np.mean(class_data, axis=0)
var_class[:] = np.var(np.vstack((data_train_pca, data_test_pca)), axis=0)

# normalize the data after PCA in order the variance equals one
data_train_pca_normed = np.zeros_like(data_train_pca)
data_test_pca_normed = np.zeros_like(data_test_pca)
data_pca_normed = np.zeros_like(np.vstack((data_train_pca, data_test_pca)))
for idx_class in range(num_class):
    for idx_train_per_class in range(num_train_per_class):
        data_train_pca_normed[num_train_per_class * idx_class + idx_train_per_class,:] = (data_train_pca[num_train_per_class * idx_class + idx_train_per_class, :] - mean_class[idx_class, :]) / np.sqrt(var_class[:]) + mean_class[idx_class, :]
    for idx_test_per_class in range(num_test_per_class):
        data_test_pca_normed[num_test_per_class * idx_class + idx_test_per_class, :] = (data_test_pca[num_test_per_class * idx_class + idx_test_per_class, :] - mean_class[idx_class, :]) / np.sqrt(var_class[:]) + mean_class[idx_class, :]
    class_data_train_normed = data_train_pca_normed[num_train_per_class * idx_class:num_train_per_class * (idx_class + 1), :]
    class_data_test_normed = data_test_pca_normed[num_test_per_class * idx_class:num_test_per_class * (idx_class + 1), :]
    class_data_normed = np.vstack((class_data_train_normed, class_data_test_normed))
    mean_class[idx_class, :] = np.mean(class_data_normed, axis=0)
    data_pca_normed[(num_train_per_class + num_test_per_class)*idx_class: (num_train_per_class + num_test_per_class)*(idx_class+1),:] = class_data_normed
var_class[:] = np.var(data_pca_normed, axis=0)

mean_class_dir = './save_model/save_mean_variance/mean_class_{}dim.npy'.format(PCA_dim)
var_class_dir = './save_model/save_mean_variance/var_class_{}dim.npy'.format(PCA_dim)
data_test_pca_normed_dir = './save_model/save_mean_variance/data_test_pca_normed_{}dim.npy'.format(PCA_dim)
label_test_dir = './save_model/save_mean_variance/label_test_{}dim.npy'.format(PCA_dim)

np.save(data_test_pca_normed_dir, data_test_pca_normed)
np.save(label_test_dir, label_test)
np.save(mean_class_dir, mean_class)
np.save(var_class_dir, var_class)


# save svm model
svm_model = './save_model/svm_model_{}dimension.pkl'.format(PCA_dim)
if os.path.exists(svm_model) is False:
    # SVM
    classifier = svm.SVC(C=1, gamma=0.001)
    print('SVM training start')
    classifier.fit(data_train_pca_normed, label_train)
    with open(svm_model, 'wb') as file:
        pickle.dump(classifier, file)
else:
    with open(svm_model, 'rb') as file:
        classifier = pickle.load(file)

# save mlp model
mlp_model = './save_model/mlp_model_{}dimension.pkl'.format(PCA_dim)
if os.path.exists(mlp_model) is False:
    # mlp
    mlp = MLPClassifier(hidden_layer_sizes=(80, 40), activation='relu', solver='adam', max_iter=160)
    print('MLP training start')
    mlp.fit(data_train_pca_normed, label_train)
    with open(mlp_model, 'wb') as file:
        pickle.dump(mlp, file)
else:
    with open(mlp_model, 'rb') as file:
        mlp = pickle.load(file)