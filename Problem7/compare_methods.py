import adam, adaGrad, adadelta, RMSProp, nadam
import matplotlib.pyplot as plt
import numpy as np

def compare():
    adam_result = adam.optimize()
    ag_result = adaGrad.optimize()
    ad_result = adadelta.optimize()
    rmsp_result = RMSProp.optimize()
    na_result = nadam.optimize()

    plt.yscale('log')
    plt.plot(adam_result, 'rs-', markersize=1, linewidth=0.5, label="Adam") # better
    plt.plot(ag_result, 'bs-', markersize=1, linewidth=0.5, label="AdaGrad")
    plt.plot(ad_result, 'cs-', markersize=1, linewidth=0.5, label="AdaDelta")
    plt.plot(rmsp_result, 'gs-', markersize=1, linewidth=0.5, label="RMS Prop") # better
    plt.plot(na_result, 'ks-', markersize=1, linewidth=0.5, label="Nadam")
    plt.legend()
    #plt.savefig("compared.png")
    #plt.savefig("compared.eps")
    plt.show()
    return

if __name__ == '__main__':
    compare()

print('module name：{}'.format(__name__))
