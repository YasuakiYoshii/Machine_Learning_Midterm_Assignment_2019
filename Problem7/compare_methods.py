import adam, adaGrad, adadelta, RMSProp, nadam
import matplotlib.pyplot as plt
import numpy as np
from common import diff_func

def compare():
    adam_result = diff_func(adam.optimize())
    ag_result = diff_func(adaGrad.optimize())
    ad_result = diff_func(adadelta.optimize())
    rmsp_result = diff_func(RMSProp.optimize())
    na_result = diff_func(nadam.optimize())

    plt.yscale('log')
    plt.plot(adam_result, 'rs-', markersize=1, linewidth=0.5, label="Adam") # better
    plt.plot(ag_result, 'bs-', markersize=1, linewidth=0.5, label="AdaGrad")
    plt.plot(ad_result, 'cs-', markersize=1, linewidth=0.5, label="AdaDelta")
    plt.plot(rmsp_result, 'gs-', markersize=1, linewidth=0.5, label="RMS Prop") # better
    plt.plot(na_result, 'ks-', markersize=1, linewidth=0.5, label="Nadam")
    plt.legend()
    plt.savefig("compared.png")
    plt.savefig("compared.eps")
    plt.savefig("compared.pdf")
    plt.show()
    return

if __name__ == '__main__':
    compare()

print('module name：{}'.format(__name__))
