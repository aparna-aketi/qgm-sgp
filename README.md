# Quasi Global Momentum (QGM) combined with Stochastic Gradient Push (SGP)

This implementation modifies the [facebook's SGP codebase](https://github.com/facebookresearch/stochastic_gradient_push) to include quasi global momentum (QGM) updates for heterogenous data distribution. The modified algorithm works for both directed and unidirected graphs with time varying structures in a given decelentralized setup. 

## References
1. Lin, T., Karimireddy, S.P., Stich, S. &amp; Jaggi, M.. (2021). Quasi-global Momentum: Accelerating Decentralized Deep Learning on Heterogeneous Data. <i>Proceedings of the 38th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 139:6654-6665 [Available here](http://proceedings.mlr.press/v139/lin21c.html).
2. Assran, M., Loizou, N., Ballas, N. &amp; Rabbat, M.. (2019). Stochastic Gradient Push for Distributed Deep Learning. <i>Proceedings of the 36th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 97:344-353 [Available here](http://proceedings.mlr.press/v97/assran19a.html).
3. Aketi, Sai Aparna, Amandeep Singh, and Jan Rabaey. "Sparse-Push: Communication-& Energy-Efficient Decentralized Distributed Learning over Directed & Time-Varying Graphs with non-IID Datasets." arXiv preprint [arXiv:2102.05715](https://arxiv.org/abs/2102.05715) (2021).

