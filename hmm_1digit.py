import argparse
import logging
import numpy as np
from sklearn.neural_network import MLPClassifier


def get_data_dict(data):
    data_dict = {}
    for line in data:
        if "[" in line:
            key = line.split()[0]
            mat = []
        elif "]" in line:
            line = line.split(']')[0]
            mat.append([float(x) for x in line.split()])
            data_dict[key] = np.array(mat)
        else:
            mat.append([float(x) for x in line.split()])
    return data_dict


def get_log_const(dim, r):
    return- 0.5 * dim * np.log(2 * np.pi) - 0.5 * np.sum(np.log(r))


def logsumexp(xs):
    max = np.max(xs)
    ds = xs - max
    return max + np.log(np.exp(ds).sum())


class SingleGauss():
    def __init__(self):
        self.dim = None
        self.stats0 = None
        self.stats1 = None
        self.stats2 = None

        self.mu = None
        self.r = None
        self.logconst = None

    def train(self, data):
        data_mat = np.vstack(data)
        self.dim = data_mat.shape[1]

        self.stats0 = data_mat.shape[0]
        self.stats1 = np.sum(data_mat, axis=0)
        self.stats2 = np.sum(data_mat * data_mat, axis=0)

        # mu = sum o / T
        self.mu = self.stats1 / self.stats0
        # r = (sum o^2 - T mu^2)/T
        self.r = (self.stats2 - self.stats0 * self.mu * self.mu) / self.stats0

        # -0.5 D log(2pi) -0.5 sum (log r_d)
        self.logconst = get_log_const(self.dim, self.r)

    def loglike(self, data_mat):
        stats0 = data_mat.shape[0]
        stats1 = np.sum(data_mat, axis=0)
        stats2 = np.sum(data_mat * data_mat, axis=0)

        # ll = T log_const - 0.5 (stats2 - 2 * stats1 * mu + stats0 * mu^2)/r
        ll = stats0 * self.logconst
        ll += -0.5 * np.sum((stats2 - 2 * stats1 * self.mu + stats0 * self.mu * self.mu)/self.r)
        
        return ll


class HMMGauss():
    def __init__(self, gauss, nstate=3):
        self.hmm = []
        self.nstate = nstate
        for j in range(nstate):
            self.hmm.append(SingleGauss())

        # init
        self.pi = np.zeros(nstate)
        self.pi[0] = 1.0 # left to right
        # state trans, left to right, consider final state
        self.a = np.zeros((nstate, nstate))
        for j in range(nstate - 1):
            self.a[j, j] = 0.5
            self.a[j, j+1] = 0.5
        self.a[nstate - 1, nstate - 1] = 1.0
        # init with purturb
        for j in range(nstate):
            self.hmm[j].dim = gauss.dim
            self.hmm[j].mu = gauss.mu
            self.hmm[j].r = gauss.r
            self.hmm[j].logconst = get_log_const(gauss.dim, gauss.r)

        # stats
        self.statsa = None

        # for log(0)
        self.bigminus = -10000000

    def uniform_state(self, data_mat):
        seq = np.zeros(data_mat.shape[0], dtype=np.int)
        interval = int(data_mat.shape[0] / self.nstate)
        seq[:interval] = 0
        prev = interval
        for i in range(1, self.nstate - 1):
            seq[prev:prev+interval] = i
            prev = prev+interval
        seq[prev:] = self.nstate - 1

        return seq

    def forward(self, data_mat):
        # delta T x J
        logalpha = np.zeros((data_mat.shape[0], self.nstate))

        # t = 1 only consider j = 0, as it is left to right
        logalpha[0, 0] = np.log(self.pi[0]) + self.hmm[0].loglike(np.expand_dims(data_mat[0], axis=0))
        for i in range(1, self.nstate):
            logalpha[0, i] = self.bigminus + self.hmm[i].loglike(np.expand_dims(data_mat[0], axis=0))
        for t in range(1, data_mat.shape[0]):
            # compute Gaussian likelihood
            lls = [self.hmm[j].loglike(np.expand_dims(data_mat[t], axis=0)) for j in range(self.nstate)]
            # get logdelta
            for j in range(self.nstate):
                logalpha_all = []
                logalpha[t, j] = lls[j]
                for i in range(self.nstate):
                    if self.a[i, j] == 0.0:
                        loga = self.bigminus
                    else:
                        loga = np.log(self.a[i, j])
                    logalpha_all.append(loga + logalpha[t-1, i])
                logalpha[t, j] += logsumexp(np.array(logalpha_all))
        
        loglike = logsumexp(logalpha[data_mat.shape[0] - 1, :])

        return logalpha, loglike

    def viterbi(self, data_mat):
        # delta T x J
        logdelta = np.zeros((data_mat.shape[0], self.nstate))
        # psi T x J
        psi = np.zeros((data_mat.shape[0], self.nstate))

        # t = 1 only consider j = 0, as it is left to right
        logdelta[0, 0] = np.log(self.pi[0]) + self.hmm[0].loglike(np.expand_dims(data_mat[0], axis=0))
        for i in range(1, self.nstate):
            logdelta[0, i] = self.bigminus + self.hmm[i].loglike(np.expand_dims(data_mat[0], axis=0))
        for t in range(1, data_mat.shape[0]):
            # compute Gaussian likelihood
            lls = [self.hmm[j].loglike(np.expand_dims(data_mat[t], axis=0)) for j in range(self.nstate)]
            # get logdelta
            for j in range(self.nstate):
                logdelta_all = []
                for i in range(self.nstate):
                    if self.a[i, j] == 0.0:
                        loga = self.bigminus
                    else:
                        loga = np.log(self.a[i, j])
                    logdelta_all.append(loga + logdelta[t-1, i] + lls[j])
                logdelta[t, j] = np.max(np.array(logdelta_all))
            # get psi
            for j in range(self.nstate):
                psi_all = []
                for i in range(self.nstate):
                    if self.a[i, j] == 0.0:
                        loga = self.bigminus
                    else:
                        loga = np.log(self.a[i, j])
                    psi_all.append(loga + logdelta[t-1, i])
                psi[t, j] = np.argmax(np.array(psi_all))

        # back tracking
        seq = np.zeros(data_mat.shape[0], dtype=np.int)
        seq[data_mat.shape[0] - 1] = np.argmax(logdelta[data_mat.shape[0] - 1, :])
        for t in range(data_mat.shape[0] - 1)[::-1]:
            seq[t] = psi[t+1, seq[t+1]]

        if seq[0] != 0:
            logging.warn("???")

        logging.debug(seq)
        return seq

    def estep_viterbi(self, data, uniform=False):
        # init stats
        for j in range(self.nstate):
            self.hmm[j].stats0 = 0
            self.hmm[j].stats1 = np.zeros(self.hmm[j].dim)
            self.hmm[j].stats2 = np.zeros(self.hmm[j].dim)
        self.statsa = np.zeros((self.nstate, self.nstate))

        data_seq = []
        for data_mat in data:
            # gamma T x J
            gamma = np.zeros((data_mat.shape[0], self.nstate))
            # xi (T-1) x J
            xi = np.zeros((data_mat.shape[0] - 1, self.nstate, self.nstate))

            if uniform is True:
                # uniform state seq
                seq = self.uniform_state(data_mat)
            else:
                # get most likely state seq
                seq = self.viterbi(data_mat)
            data_seq.append(seq)

            # t = 1
            gamma[0, 0] = 1
            for i in range(self.nstate):
                self.hmm[i].stats0 += gamma[0, i]
                self.hmm[i].stats1 += gamma[0, i] * data_mat[0]
                self.hmm[i].stats2 += gamma[0, i] * data_mat[0] * data_mat[0]
            # t > 1
            for t in range(1, data_mat.shape[0]):
                gamma[t, seq[t]] = 1
                for i in range(self.nstate):
                    for j in range(self.nstate):
                        if seq[t-1] == i and seq[t] == j:
                            xi[t-1, i, j] = 1

                for i in range(self.nstate):
                    self.hmm[i].stats0 += gamma[t, i]
                    self.hmm[i].stats1 += gamma[t, i] * data_mat[t]
                    self.hmm[i].stats2 += gamma[t, i] * data_mat[t] * data_mat[t]
                    for j in range(self.nstate):
                        self.statsa[i, j] += xi[t-1, i, j]
        
        return data_seq

    def mstep(self):
        self.pi = np.zeros(self.nstate)
        self.pi[0] = 1.0

        # state transition (left to right)
        self.a = np.zeros((self.nstate, self.nstate))
        for j in range(self.nstate - 1):
            self.a[j, j] = self.statsa[j, j] / (self.statsa[j, j] + self.statsa[j, j+1])
            self.a[j, j+1] = self.statsa[j, j+1] / (self.statsa[j, j] + self.statsa[j, j+1])
        self.a[self.nstate - 1, self.nstate - 1] = 1.0

        for j in range(self.nstate):
            # mu = sum o / T
            self.hmm[j].mu = self.hmm[j].stats1 / self.hmm[j].stats0
            # r = (sum o^2 - T mu^2)/T
            self.hmm[j].r = (self.hmm[j].stats2 - self.hmm[j].stats0 * self.hmm[j].mu * self.hmm[j].mu) / self.hmm[j].stats0

            # -0.5 D log(2pi) -0.5 sum (log r_d)
            self.hmm[j].logconst = - 0.5 * self.hmm[j].dim * np.log(2 * np.pi) - 0.5 * np.sum(np.log(self.hmm[j].r))

        logging.debug("count per digit = %f", sum([self.hmm[j].stats0 for j in range(self.nstate)]))

    def loglike(self, data_mat, seq=None):
        if seq is None:
            seq = self.viterbi(data_mat)

        ll = np.log(self.pi[0]) + self.hmm[0].loglike(np.expand_dims(data_mat[0], axis=0))
        prev_state = 0
        for t in range(1, data_mat.shape[0]):
            state = seq[t]
            ll += self.hmm[state].loglike(np.expand_dims(data_mat[t], axis=0)) + np.log(self.a[prev_state, state])
            prev_state = state

        return ll

    def vit_llh(self, data_mat, p1, p2):
        # delta T x J
        logdelta = np.zeros((data_mat.shape[0], self.nstate))

        # t = 1 only consider j = 0, as it is left to right
        logdelta[0, 0] = np.log(self.pi[0]) + p1[0][0] - p2[0]
        for i in range(1, self.nstate):
            logdelta[0, i] = self.bigminus + p1[0][i] - p2[i]
        for t in range(1, data_mat.shape[0]):
            # compute Gaussian likelihood
            lls = [p1[t][j] - p2[j] for j in range(self.nstate)]
            # get logdelta
            for j in range(self.nstate):
                logdelta_all = []
                for i in range(self.nstate):
                    if self.a[i, j] == 0.0:
                        loga = self.bigminus
                    else:
                        loga = np.log(self.a[i, j])
                    logdelta_all.append(loga + logdelta[t - 1, i] + lls[j])
                logdelta[t, j] = np.max(np.array(logdelta_all))

        return np.max(logdelta[-1])

    #def forward_for_test


class MixtureGauss():
    def __init__(self, gauss, ncomp=2, eps=0.01):
        self.gmm = []
        self.ncomp = ncomp
        for n in range(ncomp):
            self.gmm.append(SingleGauss())

        # init
        self.mix = np.ones(ncomp) * (1/ncomp)
        # init with purturb
        rand_purturb = abs(np.random.randn(ncomp))
        for n in range(ncomp):
            self.gmm[n].dim = gauss.dim
            self.gmm[n].mu = gauss.mu + eps * rand_purturb[n] * np.sqrt(gauss.r)
            self.gmm[n].r = gauss.r
            self.gmm[n].logconst = get_log_const(gauss.dim, gauss.r)

    def estep(self, data):
        data_mat = np.vstack(data)
        # init stats
        for k in range(self.ncomp):
            self.gmm[k].stats0 = 0
            self.gmm[k].stats1 = np.zeros(self.gmm[k].dim)
            self.gmm[k].stats2 = np.zeros(self.gmm[k].dim)
        # gamma T x K
        gamma = np.zeros((data_mat.shape[0], self.ncomp))
        for i in range(data_mat.shape[0]):
            lls = [self.gmm[k].loglike(np.expand_dims(data_mat[i], axis=0)) + np.log(self.mix[k]) for k in range(self.ncomp)]
            lsum = logsumexp(np.array(lls))
            gamma[i,:] = np.array(np.exp(lls))/np.exp(lsum)

            for k in range(self.ncomp):
                self.gmm[k].stats0 += gamma[i, k]
                self.gmm[k].stats1 += gamma[i, k] * data_mat[i]
                self.gmm[k].stats2 += gamma[i, k] * data_mat[i] * data_mat[i]

    def mstep(self):
        summix = 0.0
        for k in range(self.ncomp):
            summix += self.gmm[k].stats0
        for k in range(self.ncomp):
            # omega
            self.mix[k] = self.gmm[k].stats0/summix
            # mu = sum o / T
            self.gmm[k].mu = self.gmm[k].stats1 / self.gmm[k].stats0
            # r = (sum o^2 - T mu^2)/T
            self.gmm[k].r = (self.gmm[k].stats2 - self.gmm[k].stats0 * self.gmm[k].mu * self.gmm[k].mu) / self.gmm[k].stats0

            # -0.5 D log(2pi) -0.5 sum (log r_d)
            self.gmm[k].logconst = - 0.5 * self.gmm[k].dim * np.log(2 * np.pi) - 0.5 * np.sum(np.log(self.gmm[k].r))

    def loglike(self, data_mat):
        ll = 0
        for i in range(data_mat.shape[0]):
            lls = [self.gmm[k].loglike(np.expand_dims(data_mat[i], axis=0)) + np.log(self.mix[k]) for k in range(self.ncomp)]
            ll += logsumexp(np.array(lls))
        
        return ll


def sg_train(digits, train_data):
    logging.info("single Gaussian training")
    model = {}
    for digit in digits:
        model[digit] = SingleGauss()

    for digit in digits:
        data = [train_data[id] for id in train_data.keys() if digit in id.split('_')[1]]

        logging.info("process %d data for digit %s", len(data), digit)

        model[digit].train(data)

    return model


def hmm_train(digits, train_data, sg_model, nstate=5, niter=10):
    logging.info("hidden Markov model training, %d states, %d iterations", nstate, niter)
    
    hmm_model = {}
    for digit in digits:
        hmm_model[digit] = HMMGauss(sg_model[digit], nstate=nstate)

    i = 0
    while i < niter:
        logging.info("iteration: %d", i)
        total_log_like = 0.0
        total_count = 0.0
        for digit in digits:
            data = [train_data[id] for id in train_data.keys() if digit in id.split('_')[1]]
            logging.info("process %d data for digit %s", len(data), digit)

            # uniform init
            if i == 0:
                data_seq = hmm_model[digit].estep_viterbi(data, uniform=True)
            else:
                data_seq = hmm_model[digit].estep_viterbi(data)

            count = sum([len(seq) for seq in data_seq])
            logging.debug("count per digit = %d", count)
            total_count += count

            hmm_model[digit].mstep()
            for data_mat, seq in zip(data, data_seq):
                #total_log_like += hmm_model[digit].loglike(data_mat, seq)
                logalpha, log_like = hmm_model[digit].forward(data_mat)
                total_log_like += log_like
        logging.info("log likelihood: %f", total_log_like)
        logging.info("total count = %d", total_count)
        i += 1

    return hmm_model

def gmm_train(digits, train_data, sg_model, ncomp=8, niter=20):
    logging.info("Gaussian mixture training, %d components, %d iterations", ncomp, niter)
    
    gmm_model = {}
    for digit in digits:
        gmm_model[digit] = MixtureGauss(sg_model[digit], ncomp=ncomp)

    i = 0
    while i < niter:
        logging.info("iteration: %d", i)
        total_log_like = 0.0
        for digit in digits:
            data = [train_data[id] for id in train_data.keys() if digit in id.split('_')[1]]
            logging.info("process %d data for digit %s", len(data), digit)
            gmm_model[digit].estep(data)
            gmm_model[digit].mstep()
            total_log_like += gmm_model[digit].loglike(np.vstack(data))
        logging.info("log likelihood: %f", total_log_like)
        i += 1

    return gmm_model


def matrix_concatenating(o_mat):
    r_mat = np.zeros(shape=(len(o_mat), 7 * 39))
    r_mat[0] = np.concatenate((o_mat[0], o_mat[0], o_mat[0], o_mat[0], o_mat[1], o_mat[2], o_mat[3]))
    r_mat[1] = np.concatenate((o_mat[0], o_mat[0], o_mat[0], o_mat[1], o_mat[2], o_mat[3], o_mat[4]))
    r_mat[2] = np.concatenate((o_mat[0], o_mat[0], o_mat[1], o_mat[2], o_mat[3], o_mat[4], o_mat[5]))
    r_mat[-1] = np.concatenate((o_mat[-4], o_mat[-3], o_mat[-2], o_mat[-1], o_mat[-1], o_mat[-1], o_mat[-1]))
    r_mat[-2] = np.concatenate((o_mat[-5], o_mat[-4], o_mat[-3], o_mat[-2], o_mat[-1], o_mat[-1], o_mat[-1]))
    r_mat[-3] = np.concatenate((o_mat[-6], o_mat[-5], o_mat[-4], o_mat[-3], o_mat[-2], o_mat[-1], o_mat[-1]))

    for n in range(3, len(o_mat) - 3):
        r_mat[n] = np.ndarray.flatten(o_mat[n - 3:n + 4, :])

    return r_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=str, help='training data')
    parser.add_argument('test', type=str, help='test data')
    parser.add_argument('--mode', type=str, default='sg',
                        choices=['sg', 'gmm', 'hmm'],
                        help='Type of models')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # set seed
    np.random.seed(777)

    # logging info
    log_format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "z", "o"]

    # read training data
    with open(args.train) as f:
        train_data = get_data_dict(f.readlines())
    # for debug
    if args.debug:
        train_data = {key: train_data[key] for key in list(train_data.keys())[:100]}
    logging.info("read %d data", len(train_data))
    train_count = 0
    for v in train_data.values():
        train_count += v.shape[0]
        dim = v.shape[1]
    logging.info("# total frames %d", train_count)

    # read test data
    with open(args.test) as f:
        test_data = get_data_dict(f.readlines())
    # for debug
    if args.debug:
        test_data = {key:test_data[key] for key in list(test_data.keys())[:100]}
    logging.info("read %d data", len(test_data))
    test_count = 0
    for v in test_data.values():
        test_count += v.shape[0]
        dim = v.shape[1]
    logging.info("# total frames %d", test_count)

    # Single Gaussian
    sg_model = sg_train(digits, train_data)

    if args.mode == 'sg':
        model = sg_model
    elif args.mode == 'hmm':
        model = hmm_train(digits, train_data, sg_model, nstate=5, niter=10)
    elif args.mode == 'gmm':
        model = gmm_train(digits, train_data, sg_model, ncomp=8, niter=20)

    ut_seq = {}
    seq_vector = np.ndarray(train_count)
    insert_flag = 0
    insert_flag_2 = 0
    expansion_data = {}
    feature_mat = np.zeros(shape=(train_count, 7*39))

    for each_id, each_mat in train_data.items():
        ut_digit = each_id[3]
        hmm_model = model[ut_digit]
        state_seq = np.array(hmm_model.viterbi(each_mat))
        if ut_digit.isdigit():
            state_seq = state_seq + (int(ut_digit) - 1) * 5
        elif ut_digit == 'o':
            state_seq = state_seq + 45
        elif ut_digit == 'z':
            state_seq = state_seq + 50

        ut_seq[each_id] = state_seq

        seq_vector[insert_flag: insert_flag + len(state_seq)] = state_seq
        insert_flag += len(state_seq)

        expansion_mat = matrix_concatenating(each_mat)
        expansion_data[each_id] = expansion_mat
        feature_mat[insert_flag_2: insert_flag_2 + len(expansion_mat), :] = expansion_mat
        insert_flag_2 += len(expansion_mat)

    pl = np.zeros(55)
    for i in range(55):
        pl[i] = np.sum(seq_vector==i)

    pl = pl/train_count

    mlp = MLPClassifier(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes = (256, 256), verbose=True, early_stopping=True)

    mlp.fit(feature_mat, seq_vector)

    # test
    total_count = 0
    correct = 0
    for key in test_data.keys():
        lls = [] 
        for digit in digits:
            if args.mode == 'hmm':
                the_pl = 0
                pro_mat = mlp.predict_log_proba(matrix_concatenating(test_data[key]))
                if digit.isdigit():
                    pro_mat = pro_mat[:, (int(digit)-1)*5: int(digit)*5]
                    the_pl = pl[(int(digit)-1)*5: int(digit)*5]
                elif digit == 'o':
                    pro_mat = pro_mat[:, 45:50]
                    the_pl = pl[45:50]
                elif digit == 'z':
                    pro_mat = pro_mat[:, 50:]
                    the_pl = pl[50:]

                ll = model[digit].vit_llh(test_data[key], pro_mat, the_pl)
            else:
                ll = model[digit].loglike(test_data[key])
            lls.append(ll)
        predict = digits[np.argmax(np.array(lls))]
        log_like = np.max(np.array(lls))

        logging.info("predict %s for utt %s (log like = %f)", predict, key, log_like)
        if predict in key.split('_')[1]:
            correct += 1
        total_count += 1

    logging.info("accuracy: %f", float(correct/total_count * 100))
