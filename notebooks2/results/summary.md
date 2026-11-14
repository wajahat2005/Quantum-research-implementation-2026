# notebooks2 summary — paper methods × diabetes datasets

Positive class: pima: diabetes; sylhet: diabetes (early stage); cdc: diabetes or prediabetes (self-reported); hosp130: readmitted within 30 days

## Clean protocol (mean over seeds × folds)

|                                                                  |   balanced_accuracy |   recall |   specificity |   roc_auc |   accuracy |   majority_floor |
|:-----------------------------------------------------------------|--------------------:|---------:|--------------:|----------:|-----------:|-----------------:|
| ('cdc', 'Kiran: RBF-SVM (same 4 PCs, balanced)')                 |               0.727 |    0.786 |         0.669 |     0.798 |      0.686 |            0.86  |
| ('cdc', 'Kiran: Logistic Regression (same 4 PCs, balanced)')     |               0.721 |    0.72  |         0.722 |     0.8   |      0.723 |            0.86  |
| ('cdc', 'Kotha: DiaBeats RF (threshold by inner CV)')            |               0.721 |    0.748 |         0.694 |     0.796 |      0.701 |            0.86  |
| ('cdc', 'Kiran: HQCAN VQC (4q, 2 layers)')                       |               0.715 |    0.714 |         0.716 |     0.786 |      0.716 |            0.86  |
| ('cdc', 'Young: QISICGM ensemble (+GMM, train-fold only)')       |               0.609 |    0.276 |         0.943 |     0.784 |      0.85  |            0.86  |
| ('cdc', 'Zhang: LightGBM + GIF weights')                         |               0.598 |    0.238 |         0.957 |     0.805 |      0.857 |            0.86  |
| ('cdc', 'Zhang: LightGBM')                                       |               0.591 |    0.212 |         0.969 |     0.812 |      0.863 |            0.86  |
| ('cdc', 'Upama: RF (DML top-10)')                                |               0.584 |    0.206 |         0.961 |     0.769 |      0.856 |            0.86  |
| ('cdc', 'Daga: Logistic Regression')                             |               0.577 |    0.18  |         0.974 |     0.818 |      0.863 |            0.86  |
| ('cdc', 'Kotha: Logistic Regression')                            |               0.577 |    0.18  |         0.974 |     0.818 |      0.863 |            0.86  |
| ('cdc', 'Sodar: Logistic Regression')                            |               0.577 |    0.18  |         0.974 |     0.818 |      0.863 |            0.86  |
| ('cdc', 'Young: QISICGM ensemble (no augmentation)')             |               0.572 |    0.175 |         0.97  |     0.798 |      0.858 |            0.86  |
| ('cdc', 'Kotha: DiaBeats RF')                                    |               0.559 |    0.138 |         0.981 |     0.796 |      0.863 |            0.86  |
| ('cdc', 'Daga: OOF-Stacking (LR meta)')                          |               0.558 |    0.133 |         0.982 |     0.756 |      0.863 |            0.86  |
| ('cdc', 'Upama: RF (all features)')                              |               0.553 |    0.124 |         0.981 |     0.796 |      0.861 |            0.86  |
| ('cdc', 'Daga: Random Forest')                                   |               0.551 |    0.12  |         0.983 |     0.801 |      0.862 |            0.86  |
| ('cdc', 'Daga: RBF-SVM')                                         |               0.531 |    0.071 |         0.99  |     0.75  |      0.862 |            0.86  |
| ('cdc', 'Sodar: RBF-SVM')                                        |               0.531 |    0.071 |         0.99  |     0.75  |      0.862 |            0.86  |
| ('cdc', 'Yadav: RBF-SVM')                                        |               0.531 |    0.071 |         0.99  |     0.75  |      0.862 |            0.86  |
| ('cdc', 'Yadav: ZZ-QSVM (full) bw=inner-CV')                     |               0.527 |    0.062 |         0.991 |     0.7   |      0.861 |            0.86  |
| ('cdc', 'Yadav: ZZ-QSVM (linear) bw=inner-CV')                   |               0.52  |    0.049 |         0.992 |     0.704 |      0.86  |            0.86  |
| ('cdc', 'Yadav: Z-QSVM bw=1')                                    |               0.513 |    0.032 |         0.995 |     0.708 |      0.86  |            0.86  |
| ('cdc', 'Yadav: Z-QSVM bw=inner-CV')                             |               0.513 |    0.032 |         0.994 |     0.7   |      0.86  |            0.86  |
| ('cdc', 'Daga: ZZ-QSVM')                                         |               0.506 |    0.014 |         0.998 |     0.684 |      0.86  |            0.86  |
| ('cdc', 'Yadav: ZZ-QSVM (linear) bw=1')                          |               0.506 |    0.014 |         0.998 |     0.684 |      0.86  |            0.86  |
| ('cdc', 'Upama: Angle-QSVM (DML top-10)')                        |               0.501 |    0.004 |         0.998 |     0.729 |      0.859 |            0.86  |
| ('cdc', 'Yadav: ZZ-QSVM (full) bw=1')                            |               0.501 |    0.001 |         1     |     0.635 |      0.861 |            0.86  |
| ('cdc', 'Daga: Amplitude-QSVM')                                  |               0.5   |    0     |         1     |     0.692 |      0.86  |            0.86  |
| ('cdc', 'Daga: Angle-QSVM')                                      |               0.5   |    0     |         1     |     0.717 |      0.86  |            0.86  |
| ('cdc', 'Daga: Hard-Voting (4 kernels)')                         |               0.5   |    0     |         1     |     0.534 |      0.86  |            0.86  |
| ('cdc', 'Daga: Majority class')                                  |               0.5   |    0     |         1     |     0.5   |      0.86  |            0.86  |
| ('cdc', 'Kiran: Angle-QSVM (same 4 PCs)')                        |               0.5   |    0     |         1     |     0.708 |      0.86  |            0.86  |
| ('cdc', 'Sodar: Amplitude-QSVM (MinMax, paper)')                 |               0.5   |    0     |         1     |     0.769 |      0.86  |            0.86  |
| ('cdc', 'Sodar: Amplitude-QSVM (standardised)')                  |               0.5   |    0     |         1     |     0.692 |      0.86  |            0.86  |
| ('cdc', 'Sodar: Angle-QSVM')                                     |               0.5   |    0     |         1     |     0.717 |      0.86  |            0.86  |
| ('cdc', 'Sodar: Majority class')                                 |               0.5   |    0     |         1     |     0.5   |      0.86  |            0.86  |
| ('cdc', 'Upama: Linear SVM (DML top-10)')                        |               0.5   |    0     |         1     |     0.639 |      0.86  |            0.86  |
| ('hosp130', 'Kiran: Logistic Regression (same 4 PCs, balanced)') |               0.542 |    0.527 |         0.558 |     0.567 |      0.555 |            0.912 |
| ('hosp130', 'Kotha: DiaBeats RF (threshold by inner CV)')        |               0.538 |    0.437 |         0.64  |     0.572 |      0.622 |            0.912 |
| ('hosp130', 'Kiran: HQCAN VQC (4q, 2 layers)')                   |               0.536 |    0.497 |         0.576 |     0.553 |      0.569 |            0.912 |
| ('hosp130', 'Kiran: RBF-SVM (same 4 PCs, balanced)')             |               0.532 |    0.453 |         0.611 |     0.543 |      0.597 |            0.912 |
| ('hosp130', 'Daga: Logistic Regression')                         |               0.503 |    0.009 |         0.996 |     0.6   |      0.909 |            0.912 |
| ('hosp130', 'Kotha: Logistic Regression')                        |               0.503 |    0.009 |         0.996 |     0.6   |      0.909 |            0.912 |
| ('hosp130', 'Sodar: Logistic Regression')                        |               0.503 |    0.009 |         0.996 |     0.6   |      0.909 |            0.912 |
| ('hosp130', 'Upama: RF (DML top-10)')                            |               0.501 |    0.013 |         0.989 |     0.563 |      0.904 |            0.912 |
| ('hosp130', 'Zhang: LightGBM')                                   |               0.501 |    0.004 |         0.997 |     0.581 |      0.91  |            0.912 |
| ('hosp130', 'Daga: Amplitude-QSVM')                              |               0.5   |    0     |         1     |     0.519 |      0.912 |            0.912 |
| ('hosp130', 'Daga: Angle-QSVM')                                  |               0.5   |    0     |         1     |     0.497 |      0.912 |            0.912 |
| ('hosp130', 'Daga: Hard-Voting (4 kernels)')                     |               0.5   |    0     |         1     |     0.5   |      0.912 |            0.912 |
| ('hosp130', 'Daga: Majority class')                              |               0.5   |    0     |         1     |     0.5   |      0.912 |            0.912 |
| ('hosp130', 'Daga: OOF-Stacking (LR meta)')                      |               0.5   |    0     |         1     |     0.512 |      0.912 |            0.912 |
| ('hosp130', 'Daga: RBF-SVM')                                     |               0.5   |    0     |         1     |     0.548 |      0.912 |            0.912 |
| ('hosp130', 'Daga: Random Forest')                               |               0.5   |    0     |         0.999 |     0.583 |      0.912 |            0.912 |
| ('hosp130', 'Daga: ZZ-QSVM')                                     |               0.5   |    0     |         1     |     0.535 |      0.912 |            0.912 |
| ('hosp130', 'Kiran: Angle-QSVM (same 4 PCs)')                    |               0.5   |    0     |         1     |     0.52  |      0.912 |            0.912 |
| ('hosp130', 'Kotha: DiaBeats RF')                                |               0.5   |    0     |         1     |     0.572 |      0.912 |            0.912 |
| ('hosp130', 'Sodar: Amplitude-QSVM (MinMax, paper)')             |               0.5   |    0     |         1     |     0.506 |      0.912 |            0.912 |
| ('hosp130', 'Sodar: Amplitude-QSVM (standardised)')              |               0.5   |    0     |         1     |     0.519 |      0.912 |            0.912 |
| ('hosp130', 'Sodar: Angle-QSVM')                                 |               0.5   |    0     |         1     |     0.497 |      0.912 |            0.912 |
| ('hosp130', 'Sodar: Majority class')                             |               0.5   |    0     |         1     |     0.5   |      0.912 |            0.912 |
| ('hosp130', 'Sodar: RBF-SVM')                                    |               0.5   |    0     |         1     |     0.548 |      0.912 |            0.912 |
| ('hosp130', 'Upama: Angle-QSVM (DML top-10)')                    |               0.5   |    0     |         1     |     0.493 |      0.912 |            0.912 |
| ('hosp130', 'Upama: Linear SVM (DML top-10)')                    |               0.5   |    0     |         1     |     0.498 |      0.912 |            0.912 |
| ('hosp130', 'Upama: RF (all features)')                          |               0.5   |    0     |         1     |     0.574 |      0.912 |            0.912 |
| ('hosp130', 'Yadav: RBF-SVM')                                    |               0.5   |    0     |         1     |     0.548 |      0.912 |            0.912 |
| ('hosp130', 'Yadav: Z-QSVM bw=1')                                |               0.5   |    0     |         1     |     0.518 |      0.912 |            0.912 |
| ('hosp130', 'Yadav: Z-QSVM bw=inner-CV')                         |               0.5   |    0     |         1     |     0.497 |      0.912 |            0.912 |
| ('hosp130', 'Yadav: ZZ-QSVM (full) bw=1')                        |               0.5   |    0     |         1     |     0.5   |      0.912 |            0.912 |
| ('hosp130', 'Yadav: ZZ-QSVM (full) bw=inner-CV')                 |               0.5   |    0     |         1     |     0.491 |      0.912 |            0.912 |
| ('hosp130', 'Yadav: ZZ-QSVM (linear) bw=1')                      |               0.5   |    0     |         1     |     0.535 |      0.912 |            0.912 |
| ('hosp130', 'Yadav: ZZ-QSVM (linear) bw=inner-CV')               |               0.5   |    0     |         1     |     0.504 |      0.912 |            0.912 |
| ('hosp130', 'Zhang: LightGBM + GIF weights')                     |               0.5   |    0.008 |         0.992 |     0.609 |      0.906 |            0.912 |
| ('hosp130', 'Young: QISICGM ensemble (no augmentation)')         |               0.499 |    0.006 |         0.993 |     0.548 |      0.906 |            0.912 |
| ('hosp130', 'Young: QISICGM ensemble (+GMM, train-fold only)')   |               0.498 |    0.01  |         0.986 |     0.548 |      0.9   |            0.912 |
| ('pima', 'Kotha: DiaBeats RF (threshold by inner CV)')           |               0.741 |    0.76  |         0.722 |     0.825 |      0.735 |            0.651 |
| ('pima', 'Young: QISICGM ensemble (no augmentation)')            |               0.735 |    0.616 |         0.855 |     0.827 |      0.771 |            0.651 |
| ('pima', 'Upama: RF (DML top-7)')                                |               0.729 |    0.596 |         0.862 |     0.826 |      0.769 |            0.651 |
| ('pima', 'Daga: Logistic Regression')                            |               0.726 |    0.571 |         0.881 |     0.832 |      0.773 |            0.651 |
| ('pima', 'Kotha: Logistic Regression')                           |               0.726 |    0.571 |         0.881 |     0.832 |      0.773 |            0.651 |
| ('pima', 'Sodar: Logistic Regression')                           |               0.726 |    0.571 |         0.881 |     0.832 |      0.773 |            0.651 |
| ('pima', 'Kotha: DiaBeats RF')                                   |               0.725 |    0.599 |         0.851 |     0.825 |      0.763 |            0.651 |
| ('pima', 'Upama: Linear SVM (DML top-7)')                        |               0.723 |    0.568 |         0.877 |     0.832 |      0.77  |            0.651 |
| ('pima', 'Zhang: LightGBM + GIF weights')                        |               0.723 |    0.591 |         0.855 |     0.832 |      0.763 |            0.651 |
| ('pima', 'Daga: Random Forest')                                  |               0.72  |    0.586 |         0.853 |     0.824 |      0.76  |            0.651 |
| ('pima', 'Upama: Angle-QSVM (DML top-7)')                        |               0.72  |    0.557 |         0.883 |     0.836 |      0.77  |            0.651 |
| ('pima', 'Upama: RF (all features)')                             |               0.72  |    0.586 |         0.854 |     0.823 |      0.76  |            0.651 |
| ('pima', 'Daga: OOF-Stacking (LR meta)')                         |               0.716 |    0.567 |         0.865 |     0.83  |      0.761 |            0.651 |
| ('pima', 'Yadav: ZZ-QSVM (linear) bw=inner-CV')                  |               0.715 |    0.556 |         0.874 |     0.823 |      0.763 |            0.651 |
| ('pima', 'Yadav: ZZ-QSVM (full) bw=inner-CV')                    |               0.714 |    0.566 |         0.862 |     0.803 |      0.759 |            0.651 |
| ('pima', 'Daga: RBF-SVM')                                        |               0.712 |    0.556 |         0.869 |     0.822 |      0.76  |            0.651 |
| ('pima', 'Sodar: RBF-SVM')                                       |               0.712 |    0.556 |         0.869 |     0.822 |      0.76  |            0.651 |
| ('pima', 'Yadav: RBF-SVM')                                       |               0.712 |    0.556 |         0.869 |     0.822 |      0.76  |            0.651 |
| ('pima', 'Yadav: Z-QSVM bw=inner-CV')                            |               0.712 |    0.54  |         0.885 |     0.832 |      0.764 |            0.651 |
| ('pima', 'Young: QISICGM ensemble (+GMM, train-fold only)')      |               0.712 |    0.876 |         0.549 |     0.809 |      0.663 |            0.651 |
| ('pima', 'Zhang: LightGBM')                                      |               0.712 |    0.589 |         0.835 |     0.811 |      0.749 |            0.651 |
| ('pima', 'Daga: Angle-QSVM')                                     |               0.711 |    0.552 |         0.87  |     0.832 |      0.759 |            0.651 |
| ('pima', 'Sodar: Angle-QSVM')                                    |               0.711 |    0.552 |         0.87  |     0.832 |      0.759 |            0.651 |
| ('pima', 'Kiran: Logistic Regression (same 4 PCs, balanced)')    |               0.71  |    0.693 |         0.727 |     0.786 |      0.715 |            0.651 |
| ('pima', 'Kiran: RBF-SVM (same 4 PCs, balanced)')                |               0.708 |    0.738 |         0.678 |     0.788 |      0.699 |            0.651 |
| ('pima', 'Yadav: Z-QSVM bw=1')                                   |               0.7   |    0.533 |         0.868 |     0.789 |      0.751 |            0.651 |
| ('pima', 'Kiran: HQCAN VQC (4q, 2 layers)')                      |               0.693 |    0.691 |         0.695 |     0.767 |      0.694 |            0.651 |
| ('pima', 'Kiran: Angle-QSVM (same 4 PCs)')                       |               0.668 |    0.452 |         0.884 |     0.792 |      0.733 |            0.651 |
| ('pima', 'Sodar: Amplitude-QSVM (MinMax, paper)')                |               0.548 |    0.167 |         0.93  |     0.748 |      0.664 |            0.651 |
| ('pima', 'Daga: Hard-Voting (4 kernels)')                        |               0.546 |    0.128 |         0.964 |     0.734 |      0.672 |            0.651 |
| ('pima', 'Daga: Amplitude-QSVM')                                 |               0.53  |    0.138 |         0.921 |     0.657 |      0.648 |            0.651 |
| ('pima', 'Sodar: Amplitude-QSVM (standardised)')                 |               0.53  |    0.138 |         0.921 |     0.657 |      0.648 |            0.651 |
| ('pima', 'Daga: ZZ-QSVM')                                        |               0.525 |    0.082 |         0.967 |     0.668 |      0.658 |            0.651 |
| ('pima', 'Yadav: ZZ-QSVM (linear) bw=1')                         |               0.525 |    0.082 |         0.967 |     0.668 |      0.658 |            0.651 |
| ('pima', 'Daga: Majority class')                                 |               0.5   |    0     |         1     |     0.5   |      0.651 |            0.651 |
| ('pima', 'Sodar: Majority class')                                |               0.5   |    0     |         1     |     0.5   |      0.651 |            0.651 |
| ('pima', 'Yadav: ZZ-QSVM (full) bw=1')                           |               0.5   |    0     |         1     |     0.606 |      0.651 |            0.651 |
| ('sylhet', 'Daga: Random Forest')                                |               0.925 |    0.943 |         0.908 |     0.972 |      0.93  |            0.615 |
| ('sylhet', 'Daga: OOF-Stacking (LR meta)')                       |               0.923 |    0.929 |         0.917 |     0.969 |      0.924 |            0.615 |
| ('sylhet', 'Daga: RBF-SVM')                                      |               0.923 |    0.94  |         0.905 |     0.972 |      0.928 |            0.615 |
| ('sylhet', 'Sodar: RBF-SVM')                                     |               0.923 |    0.94  |         0.905 |     0.972 |      0.928 |            0.615 |
| ('sylhet', 'Yadav: RBF-SVM')                                     |               0.923 |    0.94  |         0.905 |     0.972 |      0.928 |            0.615 |
| ('sylhet', 'Kotha: DiaBeats RF')                                 |               0.92  |    0.937 |         0.902 |     0.971 |      0.925 |            0.615 |
| ('sylhet', 'Upama: RF (all features)')                           |               0.92  |    0.935 |         0.906 |     0.971 |      0.924 |            0.615 |
| ('sylhet', 'Young: QISICGM ensemble (no augmentation)')          |               0.919 |    0.937 |         0.902 |     0.974 |      0.925 |            0.615 |
| ('sylhet', 'Young: QISICGM ensemble (+GMM, train-fold only)')    |               0.916 |    0.954 |         0.878 |     0.975 |      0.925 |            0.615 |
| ('sylhet', 'Kotha: DiaBeats RF (threshold by inner CV)')         |               0.907 |    0.91  |         0.904 |     0.971 |      0.907 |            0.615 |
| ('sylhet', 'Upama: RF (DML top-10)')                             |               0.905 |    0.948 |         0.862 |     0.961 |      0.918 |            0.615 |
| ('sylhet', 'Sodar: Amplitude-QSVM (MinMax, paper)')              |               0.903 |    0.951 |         0.856 |     0.969 |      0.915 |            0.615 |
| ('sylhet', 'Kiran: RBF-SVM (same 4 PCs, balanced)')              |               0.9   |    0.906 |         0.893 |     0.938 |      0.901 |            0.615 |
| ('sylhet', 'Kiran: Angle-QSVM (same 4 PCs)')                     |               0.898 |    0.913 |         0.883 |     0.949 |      0.901 |            0.615 |
| ('sylhet', 'Daga: Hard-Voting (4 kernels)')                      |               0.897 |    0.939 |         0.854 |     0.932 |      0.907 |            0.615 |
| ('sylhet', 'Upama: Angle-QSVM (DML top-10)')                     |               0.895 |    0.929 |         0.861 |     0.954 |      0.902 |            0.615 |
| ('sylhet', 'Yadav: Z-QSVM bw=inner-CV')                          |               0.887 |    0.886 |         0.887 |     0.959 |      0.885 |            0.615 |
| ('sylhet', 'Daga: Logistic Regression')                          |               0.886 |    0.914 |         0.857 |     0.952 |      0.893 |            0.615 |
| ('sylhet', 'Kotha: Logistic Regression')                         |               0.886 |    0.914 |         0.857 |     0.952 |      0.893 |            0.615 |
| ('sylhet', 'Sodar: Logistic Regression')                         |               0.886 |    0.914 |         0.857 |     0.952 |      0.893 |            0.615 |
| ('sylhet', 'Zhang: LightGBM')                                    |               0.885 |    0.91  |         0.861 |     0.956 |      0.892 |            0.615 |
| ('sylhet', 'Daga: Angle-QSVM')                                   |               0.882 |    0.918 |         0.846 |     0.958 |      0.888 |            0.615 |
| ('sylhet', 'Sodar: Angle-QSVM')                                  |               0.882 |    0.918 |         0.846 |     0.958 |      0.888 |            0.615 |
| ('sylhet', 'Kiran: Logistic Regression (same 4 PCs, balanced)')  |               0.874 |    0.86  |         0.889 |     0.944 |      0.871 |            0.615 |
| ('sylhet', 'Zhang: LightGBM + GIF weights')                      |               0.869 |    0.861 |         0.877 |     0.941 |      0.869 |            0.615 |
| ('sylhet', 'Upama: Linear SVM (DML top-10)')                     |               0.863 |    0.875 |         0.85  |     0.948 |      0.864 |            0.615 |
| ('sylhet', 'Yadav: ZZ-QSVM (full) bw=inner-CV')                  |               0.85  |    0.92  |         0.78  |     0.948 |      0.868 |            0.615 |
| ('sylhet', 'Yadav: ZZ-QSVM (linear) bw=inner-CV')                |               0.848 |    0.878 |         0.817 |     0.942 |      0.856 |            0.615 |
| ('sylhet', 'Daga: Amplitude-QSVM')                               |               0.798 |    0.903 |         0.693 |     0.9   |      0.823 |            0.615 |
| ('sylhet', 'Sodar: Amplitude-QSVM (standardised)')               |               0.798 |    0.903 |         0.693 |     0.9   |      0.823 |            0.615 |
| ('sylhet', 'Yadav: Z-QSVM bw=1')                                 |               0.75  |    0.924 |         0.576 |     0.857 |      0.79  |            0.615 |
| ('sylhet', 'Kiran: HQCAN VQC (4q, 2 layers)')                    |               0.736 |    0.76  |         0.712 |     0.829 |      0.74  |            0.615 |
| ('sylhet', 'Daga: ZZ-QSVM')                                      |               0.653 |    0.985 |         0.322 |     0.792 |      0.733 |            0.615 |
| ('sylhet', 'Yadav: ZZ-QSVM (linear) bw=1')                       |               0.653 |    0.985 |         0.322 |     0.792 |      0.733 |            0.615 |
| ('sylhet', 'Yadav: ZZ-QSVM (full) bw=1')                         |               0.637 |    0.985 |         0.29  |     0.707 |      0.721 |            0.615 |
| ('sylhet', 'Daga: Majority class')                               |               0.5   |    1     |         0     |     0.5   |      0.619 |            0.615 |
| ('sylhet', 'Sodar: Majority class')                              |               0.5   |    1     |         0     |     0.5   |      0.619 |            0.615 |

## Paper vs clean protocol (accuracy)

|                                                                  |   clean |   paper |   gap (paper - clean) |
|:-----------------------------------------------------------------|--------:|--------:|----------------------:|
| ('cdc', 'Daga: Amplitude-QSVM')                                  |   0.86  |   0.86  |                 0     |
| ('cdc', 'Daga: Angle-QSVM')                                      |   0.86  |   0.86  |                 0     |
| ('cdc', 'Daga: Hard-Voting (4 kernels)')                         |   0.86  |   0.86  |                 0     |
| ('cdc', 'Daga: Logistic Regression')                             |   0.863 |   0.857 |                -0.005 |
| ('cdc', 'Daga: Majority class')                                  |   0.86  |   0.86  |                 0     |
| ('cdc', 'Daga: OOF-Stacking (LR meta)')                          |   0.863 |   0.864 |                 0.001 |
| ('cdc', 'Daga: RBF-SVM')                                         |   0.862 |   0.862 |                 0.001 |
| ('cdc', 'Daga: Random Forest')                                   |   0.862 |   0.866 |                 0.004 |
| ('cdc', 'Daga: ZZ-QSVM')                                         |   0.86  |   0.86  |                 0     |
| ('cdc', 'Kiran: Angle-QSVM (same 4 PCs)')                        |   0.86  |   0.86  |                -0     |
| ('cdc', 'Kiran: HQCAN VQC (4q, 2 layers)')                       |   0.716 |   0.732 |                 0.016 |
| ('cdc', 'Kiran: Logistic Regression (same 4 PCs, balanced)')     |   0.723 |   0.735 |                 0.012 |
| ('cdc', 'Kiran: RBF-SVM (same 4 PCs, balanced)')                 |   0.686 |   0.705 |                 0.019 |
| ('cdc', 'Kotha: DiaBeats RF')                                    |   0.863 |   0.875 |                 0.012 |
| ('cdc', 'Kotha: Logistic Regression')                            |   0.863 |   0.865 |                 0.002 |
| ('cdc', 'Sodar: Amplitude-QSVM (MinMax, paper)')                 |   0.86  |   0.86  |                -0     |
| ('cdc', 'Sodar: Amplitude-QSVM (standardised)')                  |   0.86  |   0.86  |                -0     |
| ('cdc', 'Sodar: Angle-QSVM')                                     |   0.86  |   0.86  |                -0     |
| ('cdc', 'Sodar: Logistic Regression')                            |   0.863 |   0.855 |                -0.008 |
| ('cdc', 'Sodar: Majority class')                                 |   0.86  |   0.86  |                -0     |
| ('cdc', 'Sodar: RBF-SVM')                                        |   0.862 |   0.862 |                 0     |
| ('cdc', 'Upama: Angle-QSVM (DML top-10)')                        |   0.859 |   0.86  |                 0.001 |
| ('cdc', 'Upama: Linear SVM (DML top-10)')                        |   0.86  |   0.86  |                -0     |
| ('cdc', 'Upama: RF (DML top-10)')                                |   0.856 |   0.878 |                 0.022 |
| ('cdc', 'Upama: RF (all features)')                              |   0.861 |   0.872 |                 0.011 |
| ('cdc', 'Yadav: RBF-SVM')                                        |   0.862 |   0.868 |                 0.006 |
| ('cdc', 'Yadav: Z-QSVM bw=1')                                    |   0.86  |   0.865 |                 0.005 |
| ('cdc', 'Yadav: ZZ-QSVM (full) bw=1')                            |   0.861 |   0.86  |                -0.001 |
| ('cdc', 'Yadav: ZZ-QSVM (linear) bw=1')                          |   0.86  |   0.855 |                -0.005 |
| ('cdc', 'Zhang: LightGBM')                                       |   0.863 |   0.859 |                -0.004 |
| ('cdc', 'Zhang: LightGBM + GIF weights')                         |   0.857 |   0.857 |                 0     |
| ('hosp130', 'Daga: Amplitude-QSVM')                              |   0.912 |   0.912 |                -0     |
| ('hosp130', 'Daga: Angle-QSVM')                                  |   0.912 |   0.912 |                -0     |
| ('hosp130', 'Daga: Hard-Voting (4 kernels)')                     |   0.912 |   0.912 |                -0     |
| ('hosp130', 'Daga: Logistic Regression')                         |   0.909 |   0.908 |                -0.001 |
| ('hosp130', 'Daga: Majority class')                              |   0.912 |   0.912 |                -0     |
| ('hosp130', 'Daga: OOF-Stacking (LR meta)')                      |   0.912 |   0.912 |                -0     |
| ('hosp130', 'Daga: RBF-SVM')                                     |   0.912 |   0.912 |                -0     |
| ('hosp130', 'Daga: Random Forest')                               |   0.912 |   0.912 |                 0     |
| ('hosp130', 'Daga: ZZ-QSVM')                                     |   0.912 |   0.912 |                -0     |
| ('hosp130', 'Kiran: Angle-QSVM (same 4 PCs)')                    |   0.912 |   0.912 |                 0     |
| ('hosp130', 'Kiran: HQCAN VQC (4q, 2 layers)')                   |   0.569 |   0.542 |                -0.027 |
| ('hosp130', 'Kiran: Logistic Regression (same 4 PCs, balanced)') |   0.555 |   0.52  |                -0.035 |
| ('hosp130', 'Kiran: RBF-SVM (same 4 PCs, balanced)')             |   0.597 |   0.512 |                -0.084 |
| ('hosp130', 'Kotha: DiaBeats RF')                                |   0.912 |   0.912 |                 0.001 |
| ('hosp130', 'Kotha: Logistic Regression')                        |   0.909 |   0.908 |                -0.002 |
| ('hosp130', 'Sodar: Amplitude-QSVM (MinMax, paper)')             |   0.912 |   0.912 |                -0     |
| ('hosp130', 'Sodar: Amplitude-QSVM (standardised)')              |   0.912 |   0.912 |                -0     |
| ('hosp130', 'Sodar: Angle-QSVM')                                 |   0.912 |   0.912 |                -0     |
| ('hosp130', 'Sodar: Logistic Regression')                        |   0.909 |   0.907 |                -0.003 |
| ('hosp130', 'Sodar: Majority class')                             |   0.912 |   0.912 |                -0     |
| ('hosp130', 'Sodar: RBF-SVM')                                    |   0.912 |   0.912 |                -0     |
| ('hosp130', 'Upama: Angle-QSVM (DML top-10)')                    |   0.912 |   0.912 |                 0     |
| ('hosp130', 'Upama: Linear SVM (DML top-10)')                    |   0.912 |   0.912 |                 0     |
| ('hosp130', 'Upama: RF (DML top-10)')                            |   0.904 |   0.912 |                 0.009 |
| ('hosp130', 'Upama: RF (all features)')                          |   0.912 |   0.912 |                 0.001 |
| ('hosp130', 'Yadav: RBF-SVM')                                    |   0.912 |   0.912 |                 0     |
| ('hosp130', 'Yadav: Z-QSVM bw=1')                                |   0.912 |   0.912 |                 0     |
| ('hosp130', 'Yadav: ZZ-QSVM (full) bw=1')                        |   0.912 |   0.912 |                 0     |
| ('hosp130', 'Yadav: ZZ-QSVM (linear) bw=1')                      |   0.912 |   0.912 |                 0     |
| ('hosp130', 'Zhang: LightGBM')                                   |   0.91  |   0.911 |                 0.001 |
| ('hosp130', 'Zhang: LightGBM + GIF weights')                     |   0.906 |   0.904 |                -0.002 |
| ('pima', 'Daga: Amplitude-QSVM')                                 |   0.648 |   0.638 |                -0.01  |
| ('pima', 'Daga: Angle-QSVM')                                     |   0.759 |   0.758 |                -0.001 |
| ('pima', 'Daga: Hard-Voting (4 kernels)')                        |   0.672 |   0.667 |                -0.006 |
| ('pima', 'Daga: Logistic Regression')                            |   0.773 |   0.775 |                 0.002 |
| ('pima', 'Daga: Majority class')                                 |   0.651 |   0.651 |                 0     |
| ('pima', 'Daga: OOF-Stacking (LR meta)')                         |   0.761 |   0.759 |                -0.002 |
| ('pima', 'Daga: RBF-SVM')                                        |   0.76  |   0.76  |                 0.001 |
| ('pima', 'Daga: Random Forest')                                  |   0.76  |   0.767 |                 0.007 |
| ('pima', 'Daga: ZZ-QSVM')                                        |   0.658 |   0.663 |                 0.004 |
| ('pima', 'Kiran: Angle-QSVM (same 4 PCs)')                       |   0.733 |   0.714 |                -0.019 |
| ('pima', 'Kiran: HQCAN VQC (4q, 2 layers)')                      |   0.694 |   0.675 |                -0.018 |
| ('pima', 'Kiran: Logistic Regression (same 4 PCs, balanced)')    |   0.715 |   0.695 |                -0.02  |
| ('pima', 'Kiran: RBF-SVM (same 4 PCs, balanced)')                |   0.699 |   0.682 |                -0.017 |
| ('pima', 'Kotha: DiaBeats RF')                                   |   0.763 |   0.76  |                -0.003 |
| ('pima', 'Kotha: Logistic Regression')                           |   0.773 |   0.714 |                -0.059 |
| ('pima', 'Sodar: Amplitude-QSVM (MinMax, paper)')                |   0.664 |   0.641 |                -0.023 |
| ('pima', 'Sodar: Amplitude-QSVM (standardised)')                 |   0.648 |   0.667 |                 0.019 |
| ('pima', 'Sodar: Angle-QSVM')                                    |   0.759 |   0.745 |                -0.015 |
| ('pima', 'Sodar: Logistic Regression')                           |   0.773 |   0.745 |                -0.028 |
| ('pima', 'Sodar: Majority class')                                |   0.651 |   0.649 |                -0.002 |
| ('pima', 'Sodar: RBF-SVM')                                       |   0.76  |   0.749 |                -0.011 |
| ('pima', 'Upama: Angle-QSVM (DML top-7)')                        |   0.77  |   0.753 |                -0.016 |
| ('pima', 'Upama: Linear SVM (DML top-7)')                        |   0.77  |   0.721 |                -0.049 |
| ('pima', 'Upama: RF (DML top-7)')                                |   0.769 |   0.779 |                 0.01  |
| ('pima', 'Upama: RF (all features)')                             |   0.76  |   0.76  |                -0.001 |
| ('pima', 'Yadav: RBF-SVM')                                       |   0.76  |   0.747 |                -0.013 |
| ('pima', 'Yadav: Z-QSVM bw=1')                                   |   0.751 |   0.734 |                -0.017 |
| ('pima', 'Yadav: ZZ-QSVM (full) bw=1')                           |   0.651 |   0.649 |                -0.002 |
| ('pima', 'Yadav: ZZ-QSVM (linear) bw=1')                         |   0.658 |   0.656 |                -0.003 |
| ('pima', 'Zhang: LightGBM')                                      |   0.749 |   0.751 |                 0.002 |
| ('pima', 'Zhang: LightGBM + GIF weights')                        |   0.763 |   0.768 |                 0.005 |
| ('sylhet', 'Daga: Amplitude-QSVM')                               |   0.823 |   0.919 |                 0.096 |
| ('sylhet', 'Daga: Angle-QSVM')                                   |   0.888 |   0.948 |                 0.06  |
| ('sylhet', 'Daga: Hard-Voting (4 kernels)')                      |   0.907 |   0.967 |                 0.06  |
| ('sylhet', 'Daga: Logistic Regression')                          |   0.893 |   0.929 |                 0.035 |
| ('sylhet', 'Daga: Majority class')                               |   0.619 |   0.615 |                -0.004 |
| ('sylhet', 'Daga: OOF-Stacking (LR meta)')                       |   0.924 |   0.963 |                 0.04  |
| ('sylhet', 'Daga: RBF-SVM')                                      |   0.928 |   0.973 |                 0.045 |
| ('sylhet', 'Daga: Random Forest')                                |   0.93  |   0.981 |                 0.051 |
| ('sylhet', 'Daga: ZZ-QSVM')                                      |   0.733 |   0.933 |                 0.2   |
| ('sylhet', 'Kiran: Angle-QSVM (same 4 PCs)')                     |   0.901 |   0.952 |                 0.051 |
| ('sylhet', 'Kiran: HQCAN VQC (4q, 2 layers)')                    |   0.74  |   0.788 |                 0.048 |
| ('sylhet', 'Kiran: Logistic Regression (same 4 PCs, balanced)')  |   0.871 |   0.923 |                 0.052 |
| ('sylhet', 'Kiran: RBF-SVM (same 4 PCs, balanced)')              |   0.901 |   0.933 |                 0.032 |
| ('sylhet', 'Kotha: DiaBeats RF')                                 |   0.925 |   0.99  |                 0.066 |
| ('sylhet', 'Kotha: Logistic Regression')                         |   0.893 |   0.942 |                 0.049 |
| ('sylhet', 'Sodar: Amplitude-QSVM (MinMax, paper)')              |   0.915 |   0.949 |                 0.033 |
| ('sylhet', 'Sodar: Amplitude-QSVM (standardised)')               |   0.823 |   0.91  |                 0.087 |
| ('sylhet', 'Sodar: Angle-QSVM')                                  |   0.888 |   0.942 |                 0.054 |
| ('sylhet', 'Sodar: Logistic Regression')                         |   0.893 |   0.923 |                 0.03  |
| ('sylhet', 'Sodar: Majority class')                              |   0.619 |   0.615 |                -0.004 |
| ('sylhet', 'Sodar: RBF-SVM')                                     |   0.928 |   0.962 |                 0.033 |
| ('sylhet', 'Upama: Angle-QSVM (DML top-10)')                     |   0.902 |   0.942 |                 0.04  |
| ('sylhet', 'Upama: Linear SVM (DML top-10)')                     |   0.864 |   0.942 |                 0.078 |
| ('sylhet', 'Upama: RF (DML top-10)')                             |   0.918 |   0.981 |                 0.063 |
| ('sylhet', 'Upama: RF (all features)')                           |   0.924 |   0.99  |                 0.066 |
| ('sylhet', 'Yadav: RBF-SVM')                                     |   0.928 |   0.981 |                 0.053 |
| ('sylhet', 'Yadav: Z-QSVM bw=1')                                 |   0.79  |   0.981 |                 0.19  |
| ('sylhet', 'Yadav: ZZ-QSVM (full) bw=1')                         |   0.721 |   0.942 |                 0.222 |
| ('sylhet', 'Yadav: ZZ-QSVM (linear) bw=1')                       |   0.733 |   0.952 |                 0.219 |
| ('sylhet', 'Zhang: LightGBM')                                    |   0.892 |   0.971 |                 0.079 |
| ('sylhet', 'Zhang: LightGBM + GIF weights')                      |   0.869 |   0.962 |                 0.093 |

## Paired tests vs RBF-SVM (balanced accuracy, Nadeau-Bengio, Holm per dataset)

|     | dataset   | model                                             |   n_pairs |   mean_diff |        t |      p |   p_holm | verdict   |
|----:|:----------|:--------------------------------------------------|----------:|------------:|---------:|-------:|---------:|:----------|
|   0 | cdc       | Daga: Amplitude-QSVM                              |        15 |     -0.031  |  -2.9755 | 0.01   |   0.2306 | n.s.      |
|   1 | cdc       | Daga: Angle-QSVM                                  |        15 |     -0.0308 |  -2.9721 | 0.0101 |   0.2306 | n.s.      |
|   2 | cdc       | Daga: Hard-Voting (4 kernels)                     |        15 |     -0.0308 |  -2.9721 | 0.0101 |   0.2306 | n.s.      |
|   3 | cdc       | Daga: Logistic Regression                         |        15 |      0.0463 |   4.6514 | 0.0004 |   0.012  | better    |
|   4 | cdc       | Daga: Majority class                              |        15 |     -0.0308 |  -2.9721 | 0.0101 |   0.2306 | n.s.      |
|   5 | cdc       | Daga: OOF-Stacking (LR meta)                      |        15 |      0.0267 |   3.5698 | 0.0031 |   0.0769 | n.s.      |
|   6 | cdc       | Daga: Random Forest                               |        15 |      0.0205 |   1.636  | 0.1241 |   0.8688 | n.s.      |
|   7 | cdc       | Daga: ZZ-QSVM                                     |        15 |     -0.025  |  -2.1281 | 0.0516 |   0.5158 | n.s.      |
|   8 | cdc       | Kiran: Angle-QSVM (same 4 PCs)                    |        15 |     -0.0308 |  -2.9721 | 0.0101 |   0.2306 | n.s.      |
|   9 | cdc       | Kiran: HQCAN VQC (4q, 2 layers)                   |        15 |      0.1843 |   8.6084 | 0      |   0      | better    |
|  10 | cdc       | Kiran: Logistic Regression (same 4 PCs, balanced) |        15 |      0.1902 |   8.8711 | 0      |   0      | better    |
|  11 | cdc       | Kiran: RBF-SVM (same 4 PCs, balanced)             |        15 |      0.1967 |   8.9269 | 0      |   0      | better    |
|  12 | cdc       | Kotha: DiaBeats RF                                |        15 |      0.0286 |   2.7497 | 0.0157 |   0.2306 | n.s.      |
|  13 | cdc       | Kotha: DiaBeats RF (threshold by inner CV)        |        15 |      0.1898 |   8.5395 | 0      |   0      | better    |
|  14 | cdc       | Kotha: Logistic Regression                        |        15 |      0.0463 |   4.6514 | 0.0004 |   0.012  | better    |
|  15 | cdc       | Sodar: Amplitude-QSVM (MinMax, paper)             |        15 |     -0.0308 |  -2.9721 | 0.0101 |   0.2306 | n.s.      |
|  16 | cdc       | Sodar: Amplitude-QSVM (standardised)              |        15 |     -0.031  |  -2.9755 | 0.01   |   0.2306 | n.s.      |
|  17 | cdc       | Sodar: Angle-QSVM                                 |        15 |     -0.0308 |  -2.9721 | 0.0101 |   0.2306 | n.s.      |
|  18 | cdc       | Sodar: Logistic Regression                        |        15 |      0.0463 |   4.6514 | 0.0004 |   0.012  | better    |
|  19 | cdc       | Sodar: Majority class                             |        15 |     -0.0308 |  -2.9721 | 0.0101 |   0.2306 | n.s.      |
|  20 | cdc       | Sodar: RBF-SVM                                    |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  21 | cdc       | Upama: Angle-QSVM (DML top-10)                    |        15 |     -0.0302 |  -2.5814 | 0.0217 |   0.2392 | n.s.      |
|  22 | cdc       | Upama: Linear SVM (DML top-10)                    |        15 |     -0.0308 |  -2.9721 | 0.0101 |   0.2306 | n.s.      |
|  23 | cdc       | Upama: RF (DML top-10)                            |        15 |      0.0529 |   4.6148 | 0.0004 |   0.012  | better    |
|  24 | cdc       | Upama: RF (all features)                          |        15 |      0.022  |   1.8209 | 0.0901 |   0.7204 | n.s.      |
|  25 | cdc       | Yadav: RBF-SVM                                    |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  26 | cdc       | Yadav: Z-QSVM bw=1                                |        15 |     -0.0177 |  -1.5203 | 0.1507 |   0.8688 | n.s.      |
|  27 | cdc       | Yadav: Z-QSVM bw=inner-CV                         |        15 |     -0.0179 |  -1.5918 | 0.1338 |   0.8688 | n.s.      |
|  28 | cdc       | Yadav: ZZ-QSVM (full) bw=1                        |        15 |     -0.0302 |  -2.856  | 0.0127 |   0.2306 | n.s.      |
|  29 | cdc       | Yadav: ZZ-QSVM (full) bw=inner-CV                 |        15 |     -0.0043 |  -0.4736 | 0.6431 |   1      | n.s.      |
|  30 | cdc       | Yadav: ZZ-QSVM (linear) bw=1                      |        15 |     -0.025  |  -2.1281 | 0.0516 |   0.5158 | n.s.      |
|  31 | cdc       | Yadav: ZZ-QSVM (linear) bw=inner-CV               |        15 |     -0.0104 |  -0.901  | 0.3828 |   1      | n.s.      |
|  32 | cdc       | Young: QISICGM ensemble (+GMM, train-fold only)   |        15 |      0.0787 |   4.6493 | 0.0004 |   0.012  | better    |
|  33 | cdc       | Young: QISICGM ensemble (no augmentation)         |        15 |      0.0417 |   3.225  | 0.0061 |   0.1466 | n.s.      |
|  34 | cdc       | Zhang: LightGBM                                   |        15 |      0.0599 |   4.4719 | 0.0005 |   0.0142 | better    |
|  35 | cdc       | Zhang: LightGBM + GIF weights                     |        15 |      0.0669 |   4.0719 | 0.0011 |   0.0297 | better    |
|  36 | hosp130   | Daga: Amplitude-QSVM                              |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  37 | hosp130   | Daga: Angle-QSVM                                  |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  38 | hosp130   | Daga: Hard-Voting (4 kernels)                     |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  39 | hosp130   | Daga: Logistic Regression                         |        15 |      0.0028 |   0.6166 | 0.5474 |   1      | n.s.      |
|  40 | hosp130   | Daga: Majority class                              |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  41 | hosp130   | Daga: OOF-Stacking (LR meta)                      |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  42 | hosp130   | Daga: Random Forest                               |        15 |     -0.0003 |  -0.634  | 0.5363 |   1      | n.s.      |
|  43 | hosp130   | Daga: ZZ-QSVM                                     |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  44 | hosp130   | Kiran: Angle-QSVM (same 4 PCs)                    |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  45 | hosp130   | Kiran: HQCAN VQC (4q, 2 layers)                   |        15 |      0.0363 |   1.4114 | 0.18   |   1      | n.s.      |
|  46 | hosp130   | Kiran: Logistic Regression (same 4 PCs, balanced) |        15 |      0.0425 |   1.4745 | 0.1625 |   1      | n.s.      |
|  47 | hosp130   | Kiran: RBF-SVM (same 4 PCs, balanced)             |        15 |      0.0318 |   1.857  | 0.0845 |   1      | n.s.      |
|  48 | hosp130   | Kotha: DiaBeats RF                                |        15 |     -0.0001 |  -0.4588 | 0.6534 |   1      | n.s.      |
|  49 | hosp130   | Kotha: DiaBeats RF (threshold by inner CV)        |        15 |      0.0383 |   2.0528 | 0.0593 |   1      | n.s.      |
|  50 | hosp130   | Kotha: Logistic Regression                        |        15 |      0.0028 |   0.6166 | 0.5474 |   1      | n.s.      |
|  51 | hosp130   | Sodar: Amplitude-QSVM (MinMax, paper)             |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  52 | hosp130   | Sodar: Amplitude-QSVM (standardised)              |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  53 | hosp130   | Sodar: Angle-QSVM                                 |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  54 | hosp130   | Sodar: Logistic Regression                        |        15 |      0.0028 |   0.6166 | 0.5474 |   1      | n.s.      |
|  55 | hosp130   | Sodar: Majority class                             |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  56 | hosp130   | Sodar: RBF-SVM                                    |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  57 | hosp130   | Upama: Angle-QSVM (DML top-10)                    |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  58 | hosp130   | Upama: Linear SVM (DML top-10)                    |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  59 | hosp130   | Upama: RF (DML top-10)                            |        15 |      0.0013 |   0.2957 | 0.7718 |   1      | n.s.      |
|  60 | hosp130   | Upama: RF (all features)                          |        15 |     -0.0002 |  -0.6734 | 0.5117 |   1      | n.s.      |
|  61 | hosp130   | Yadav: RBF-SVM                                    |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  62 | hosp130   | Yadav: Z-QSVM bw=1                                |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  63 | hosp130   | Yadav: Z-QSVM bw=inner-CV                         |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  64 | hosp130   | Yadav: ZZ-QSVM (full) bw=1                        |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  65 | hosp130   | Yadav: ZZ-QSVM (full) bw=inner-CV                 |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  66 | hosp130   | Yadav: ZZ-QSVM (linear) bw=1                      |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  67 | hosp130   | Yadav: ZZ-QSVM (linear) bw=inner-CV               |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  68 | hosp130   | Young: QISICGM ensemble (+GMM, train-fold only)   |        15 |     -0.0021 |  -0.5712 | 0.5769 |   1      | n.s.      |
|  69 | hosp130   | Young: QISICGM ensemble (no augmentation)         |        15 |     -0.0006 |  -0.2341 | 0.8183 |   1      | n.s.      |
|  70 | hosp130   | Zhang: LightGBM                                   |        15 |      0.0005 |   0.207  | 0.839  |   1      | n.s.      |
|  71 | hosp130   | Zhang: LightGBM + GIF weights                     |        15 |     -0.0001 |  -0.016  | 0.9874 |   1      | n.s.      |
|  72 | pima      | Daga: Amplitude-QSVM                              |        15 |     -0.1826 |  -8.0898 | 0      |   0      | worse     |
|  73 | pima      | Daga: Angle-QSVM                                  |        15 |     -0.0012 |  -0.113  | 0.9117 |   1      | n.s.      |
|  74 | pima      | Daga: Hard-Voting (4 kernels)                     |        15 |     -0.1663 |  -8.3699 | 0      |   0      | worse     |
|  75 | pima      | Daga: Logistic Regression                         |        15 |      0.0138 |   1.2537 | 0.2305 |   1      | n.s.      |
|  76 | pima      | Daga: Majority class                              |        15 |     -0.2123 | -10.5716 | 0      |   0      | worse     |
|  77 | pima      | Daga: OOF-Stacking (LR meta)                      |        15 |      0.004  |   0.539  | 0.5984 |   1      | n.s.      |
|  78 | pima      | Daga: Random Forest                               |        15 |      0.0073 |   0.4314 | 0.6728 |   1      | n.s.      |
|  79 | pima      | Daga: ZZ-QSVM                                     |        15 |     -0.1877 |  -7.8779 | 0      |   0      | worse     |
|  80 | pima      | Kiran: Angle-QSVM (same 4 PCs)                    |        15 |     -0.0445 |  -2.069  | 0.0575 |   1      | n.s.      |
|  81 | pima      | Kiran: HQCAN VQC (4q, 2 layers)                   |        15 |     -0.0193 |  -1.1475 | 0.2704 |   1      | n.s.      |
|  82 | pima      | Kiran: Logistic Regression (same 4 PCs, balanced) |        15 |     -0.0023 |  -0.0974 | 0.9238 |   1      | n.s.      |
|  83 | pima      | Kiran: RBF-SVM (same 4 PCs, balanced)             |        15 |     -0.0046 |  -0.2095 | 0.8371 |   1      | n.s.      |
|  84 | pima      | Kotha: DiaBeats RF                                |        15 |      0.0128 |   0.7518 | 0.4646 |   1      | n.s.      |
|  85 | pima      | Kotha: DiaBeats RF (threshold by inner CV)        |        15 |      0.0287 |   1.1838 | 0.2562 |   1      | n.s.      |
|  86 | pima      | Kotha: Logistic Regression                        |        15 |      0.0138 |   1.2537 | 0.2305 |   1      | n.s.      |
|  87 | pima      | Sodar: Amplitude-QSVM (MinMax, paper)             |        15 |     -0.1638 |  -7.0987 | 0      |   0.0001 | worse     |
|  88 | pima      | Sodar: Amplitude-QSVM (standardised)              |        15 |     -0.1826 |  -8.0898 | 0      |   0      | worse     |
|  89 | pima      | Sodar: Angle-QSVM                                 |        15 |     -0.0012 |  -0.113  | 0.9117 |   1      | n.s.      |
|  90 | pima      | Sodar: Logistic Regression                        |        15 |      0.0138 |   1.2537 | 0.2305 |   1      | n.s.      |
|  91 | pima      | Sodar: Majority class                             |        15 |     -0.2123 | -10.5716 | 0      |   0      | worse     |
|  92 | pima      | Sodar: RBF-SVM                                    |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  93 | pima      | Upama: Angle-QSVM (DML top-7)                     |        15 |      0.008  |   0.6679 | 0.515  |   1      | n.s.      |
|  94 | pima      | Upama: Linear SVM (DML top-7)                     |        15 |      0.0106 |   0.8369 | 0.4167 |   1      | n.s.      |
|  95 | pima      | Upama: RF (DML top-7)                             |        15 |      0.0165 |   0.9051 | 0.3807 |   1      | n.s.      |
|  96 | pima      | Upama: RF (all features)                          |        15 |      0.0076 |   0.4541 | 0.6567 |   1      | n.s.      |
|  97 | pima      | Yadav: RBF-SVM                                    |        15 |      0      |   0      | 1      |   1      | n.s.      |
|  98 | pima      | Yadav: Z-QSVM bw=1                                |        15 |     -0.012  |  -0.7534 | 0.4637 |   1      | n.s.      |
|  99 | pima      | Yadav: Z-QSVM bw=inner-CV                         |        15 |     -0      |  -0.0047 | 0.9963 |   1      | n.s.      |
| 100 | pima      | Yadav: ZZ-QSVM (full) bw=1                        |        15 |     -0.2123 | -10.5716 | 0      |   0      | worse     |
| 101 | pima      | Yadav: ZZ-QSVM (full) bw=inner-CV                 |        15 |      0.0016 |   0.1307 | 0.8978 |   1      | n.s.      |
| 102 | pima      | Yadav: ZZ-QSVM (linear) bw=1                      |        15 |     -0.1877 |  -7.8779 | 0      |   0      | worse     |
| 103 | pima      | Yadav: ZZ-QSVM (linear) bw=inner-CV               |        15 |      0.0027 |   0.1689 | 0.8683 |   1      | n.s.      |
| 104 | pima      | Young: QISICGM ensemble (+GMM, train-fold only)   |        15 |     -0.0002 |  -0.0069 | 0.9946 |   1      | n.s.      |
| 105 | pima      | Young: QISICGM ensemble (no augmentation)         |        15 |      0.0228 |   1.611  | 0.1295 |   1      | n.s.      |
| 106 | pima      | Zhang: LightGBM                                   |        15 |     -0.0003 |  -0.0153 | 0.988  |   1      | n.s.      |
| 107 | pima      | Zhang: LightGBM + GIF weights                     |        15 |      0.0107 |   0.6807 | 0.5072 |   1      | n.s.      |
| 108 | sylhet    | Daga: Amplitude-QSVM                              |        15 |     -0.1249 |  -2.9937 | 0.0097 |   0.2902 | n.s.      |
| 109 | sylhet    | Daga: Angle-QSVM                                  |        15 |     -0.0411 |  -1.7127 | 0.1088 |   1      | n.s.      |
| 110 | sylhet    | Daga: Hard-Voting (4 kernels)                     |        15 |     -0.0259 |  -1.3525 | 0.1976 |   1      | n.s.      |
| 111 | sylhet    | Daga: Logistic Regression                         |        15 |     -0.0372 |  -2.106  | 0.0537 |   1      | n.s.      |
| 112 | sylhet    | Daga: Majority class                              |        15 |     -0.4228 | -16.6957 | 0      |   0      | worse     |
| 113 | sylhet    | Daga: OOF-Stacking (LR meta)                      |        15 |      0.0001 |   0.0086 | 0.9933 |   1      | n.s.      |
| 114 | sylhet    | Daga: Random Forest                               |        15 |      0.0025 |   0.1193 | 0.9067 |   1      | n.s.      |
| 115 | sylhet    | Daga: ZZ-QSVM                                     |        15 |     -0.2697 |  -5.864  | 0      |   0.0014 | worse     |
| 116 | sylhet    | Kiran: Angle-QSVM (same 4 PCs)                    |        15 |     -0.0245 |  -1.1829 | 0.2565 |   1      | n.s.      |
| 117 | sylhet    | Kiran: HQCAN VQC (4q, 2 layers)                   |        15 |     -0.1868 |  -2.5677 | 0.0223 |   0.6031 | n.s.      |
| 118 | sylhet    | Kiran: Logistic Regression (same 4 PCs, balanced) |        15 |     -0.0486 |  -2.0232 | 0.0626 |   1      | n.s.      |
| 119 | sylhet    | Kiran: RBF-SVM (same 4 PCs, balanced)             |        15 |     -0.0231 |  -1.352  | 0.1978 |   1      | n.s.      |
| 120 | sylhet    | Kotha: DiaBeats RF                                |        15 |     -0.0029 |  -0.1297 | 0.8986 |   1      | n.s.      |
| 121 | sylhet    | Kotha: DiaBeats RF (threshold by inner CV)        |        15 |     -0.0162 |  -0.535  | 0.601  |   1      | n.s.      |
| 122 | sylhet    | Kotha: Logistic Regression                        |        15 |     -0.0372 |  -2.106  | 0.0537 |   1      | n.s.      |
| 123 | sylhet    | Sodar: Amplitude-QSVM (MinMax, paper)             |        15 |     -0.0197 |  -1.1021 | 0.289  |   1      | n.s.      |
| 124 | sylhet    | Sodar: Amplitude-QSVM (standardised)              |        15 |     -0.1249 |  -2.9937 | 0.0097 |   0.2902 | n.s.      |
| 125 | sylhet    | Sodar: Angle-QSVM                                 |        15 |     -0.0411 |  -1.7127 | 0.1088 |   1      | n.s.      |
| 126 | sylhet    | Sodar: Logistic Regression                        |        15 |     -0.0372 |  -2.106  | 0.0537 |   1      | n.s.      |
| 127 | sylhet    | Sodar: Majority class                             |        15 |     -0.4228 | -16.6957 | 0      |   0      | worse     |
| 128 | sylhet    | Sodar: RBF-SVM                                    |        15 |      0      |   0      | 1      |   1      | n.s.      |
| 129 | sylhet    | Upama: Angle-QSVM (DML top-10)                    |        15 |     -0.0275 |  -1.3263 | 0.206  |   1      | n.s.      |
| 130 | sylhet    | Upama: Linear SVM (DML top-10)                    |        15 |     -0.0601 |  -2.4251 | 0.0294 |   0.7648 | n.s.      |
| 131 | sylhet    | Upama: RF (DML top-10)                            |        15 |     -0.0175 |  -0.6131 | 0.5496 |   1      | n.s.      |
| 132 | sylhet    | Upama: RF (all features)                          |        15 |     -0.0024 |  -0.118  | 0.9077 |   1      | n.s.      |
| 133 | sylhet    | Yadav: RBF-SVM                                    |        15 |      0      |   0      | 1      |   1      | n.s.      |
| 134 | sylhet    | Yadav: Z-QSVM bw=1                                |        15 |     -0.1728 |  -4.2781 | 0.0008 |   0.0237 | worse     |
| 135 | sylhet    | Yadav: Z-QSVM bw=inner-CV                         |        15 |     -0.0362 |  -1.4319 | 0.1741 |   1      | n.s.      |
| 136 | sylhet    | Yadav: ZZ-QSVM (full) bw=1                        |        15 |     -0.2855 |  -7.5712 | 0      |   0.0001 | worse     |
| 137 | sylhet    | Yadav: ZZ-QSVM (full) bw=inner-CV                 |        15 |     -0.0729 |  -2.8432 | 0.013  |   0.3647 | n.s.      |
| 138 | sylhet    | Yadav: ZZ-QSVM (linear) bw=1                      |        15 |     -0.2697 |  -5.864  | 0      |   0.0014 | worse     |
| 139 | sylhet    | Yadav: ZZ-QSVM (linear) bw=inner-CV               |        15 |     -0.0752 |  -2.0692 | 0.0575 |   1      | n.s.      |
| 140 | sylhet    | Young: QISICGM ensemble (+GMM, train-fold only)   |        15 |     -0.0063 |  -0.2482 | 0.8076 |   1      | n.s.      |
| 141 | sylhet    | Young: QISICGM ensemble (no augmentation)         |        15 |     -0.0033 |  -0.1562 | 0.8781 |   1      | n.s.      |
| 142 | sylhet    | Zhang: LightGBM                                   |        15 |     -0.0373 |  -1.4173 | 0.1783 |   1      | n.s.      |
| 143 | sylhet    | Zhang: LightGBM + GIF weights                     |        15 |     -0.054  |  -1.7571 | 0.1007 |   1      | n.s.      |

## Paper-protocol pooled accuracy, bootstrap 95 % CI

|     | dataset   | model                                             |    n |   accuracy |   ci_low |   ci_high |
|----:|:----------|:--------------------------------------------------|-----:|-----------:|---------:|----------:|
|   0 | cdc       | Daga: Amplitude-QSVM                              | 2000 |     0.8605 |   0.8455 |    0.8755 |
|   1 | cdc       | Daga: Angle-QSVM                                  | 2000 |     0.8605 |   0.845  |    0.8755 |
|   2 | cdc       | Daga: Hard-Voting (4 kernels)                     | 2000 |     0.8605 |   0.845  |    0.8765 |
|   3 | cdc       | Daga: Logistic Regression                         | 2000 |     0.8575 |   0.842  |    0.8725 |
|   4 | cdc       | Daga: Majority class                              | 2000 |     0.8605 |   0.845  |    0.8755 |
|   5 | cdc       | Daga: OOF-Stacking (LR meta)                      | 2000 |     0.8645 |   0.8485 |    0.879  |
|   6 | cdc       | Daga: RBF-SVM                                     | 2000 |     0.8625 |   0.848  |    0.877  |
|   7 | cdc       | Daga: Random Forest                               | 2000 |     0.8665 |   0.8515 |    0.8815 |
|   8 | cdc       | Daga: ZZ-QSVM                                     | 2000 |     0.8605 |   0.8455 |    0.8755 |
|   9 | cdc       | Kiran: Angle-QSVM (same 4 PCs)                    |  400 |     0.86   |   0.825  |    0.895  |
|  10 | cdc       | Kiran: HQCAN VQC (4q, 2 layers)                   |  400 |     0.7325 |   0.69   |    0.7725 |
|  11 | cdc       | Kiran: Logistic Regression (same 4 PCs, balanced) |  400 |     0.735  |   0.6925 |    0.7775 |
|  12 | cdc       | Kiran: RBF-SVM (same 4 PCs, balanced)             |  400 |     0.705  |   0.6624 |    0.75   |
|  13 | cdc       | Kotha: DiaBeats RF                                |  400 |     0.875  |   0.84   |    0.905  |
|  14 | cdc       | Kotha: Logistic Regression                        |  400 |     0.865  |   0.83   |    0.8975 |
|  15 | cdc       | Sodar: Amplitude-QSVM (MinMax, paper)             |  600 |     0.86   |   0.8317 |    0.8867 |
|  16 | cdc       | Sodar: Amplitude-QSVM (standardised)              |  600 |     0.86   |   0.83   |    0.8867 |
|  17 | cdc       | Sodar: Angle-QSVM                                 |  600 |     0.86   |   0.8333 |    0.8867 |
|  18 | cdc       | Sodar: Logistic Regression                        |  600 |     0.855  |   0.8283 |    0.8817 |
|  19 | cdc       | Sodar: Majority class                             |  600 |     0.86   |   0.8333 |    0.8867 |
|  20 | cdc       | Sodar: RBF-SVM                                    |  600 |     0.8617 |   0.835  |    0.8883 |
|  21 | cdc       | Upama: Angle-QSVM (DML top-10)                    |  400 |     0.86   |   0.825  |    0.8925 |
|  22 | cdc       | Upama: Linear SVM (DML top-10)                    |  400 |     0.86   |   0.825  |    0.8925 |
|  23 | cdc       | Upama: RF (DML top-10)                            |  400 |     0.8775 |   0.8449 |    0.91   |
|  24 | cdc       | Upama: RF (all features)                          |  400 |     0.8725 |   0.8399 |    0.905  |
|  25 | cdc       | Yadav: RBF-SVM                                    |  400 |     0.8675 |   0.835  |    0.9025 |
|  26 | cdc       | Yadav: Z-QSVM bw=1                                |  400 |     0.865  |   0.8324 |    0.8975 |
|  27 | cdc       | Yadav: ZZ-QSVM (full) bw=1                        |  400 |     0.86   |   0.825  |    0.8925 |
|  28 | cdc       | Yadav: ZZ-QSVM (linear) bw=1                      |  400 |     0.855  |   0.82   |    0.89   |
|  29 | cdc       | Young: QISICGM ensemble (+GMM, global)            | 7000 |     0.9574 |   0.9527 |    0.962  |
|  30 | cdc       | Zhang: LightGBM                                   | 2000 |     0.859  |   0.8425 |    0.8735 |
|  31 | cdc       | Zhang: LightGBM + GIF weights                     | 2000 |     0.857  |   0.841  |    0.872  |
|  32 | hosp130   | Daga: Amplitude-QSVM                              | 2000 |     0.912  |   0.8995 |    0.9245 |
|  33 | hosp130   | Daga: Angle-QSVM                                  | 2000 |     0.912  |   0.899  |    0.924  |
|  34 | hosp130   | Daga: Hard-Voting (4 kernels)                     | 2000 |     0.912  |   0.9    |    0.9245 |
|  35 | hosp130   | Daga: Logistic Regression                         | 2000 |     0.908  |   0.895  |    0.9205 |
|  36 | hosp130   | Daga: Majority class                              | 2000 |     0.912  |   0.8995 |    0.9245 |
|  37 | hosp130   | Daga: OOF-Stacking (LR meta)                      | 2000 |     0.912  |   0.8995 |    0.925  |
|  38 | hosp130   | Daga: RBF-SVM                                     | 2000 |     0.912  |   0.899  |    0.9245 |
|  39 | hosp130   | Daga: Random Forest                               | 2000 |     0.9115 |   0.8985 |    0.9235 |
|  40 | hosp130   | Daga: ZZ-QSVM                                     | 2000 |     0.912  |   0.899  |    0.9245 |
|  41 | hosp130   | Kiran: Angle-QSVM (same 4 PCs)                    |  400 |     0.9125 |   0.885  |    0.94   |
|  42 | hosp130   | Kiran: HQCAN VQC (4q, 2 layers)                   |  400 |     0.5425 |   0.495  |    0.5925 |
|  43 | hosp130   | Kiran: Logistic Regression (same 4 PCs, balanced) |  400 |     0.52   |   0.47   |    0.57   |
|  44 | hosp130   | Kiran: RBF-SVM (same 4 PCs, balanced)             |  400 |     0.5125 |   0.4625 |    0.56   |
|  45 | hosp130   | Kotha: DiaBeats RF                                |  400 |     0.9125 |   0.885  |    0.9375 |
|  46 | hosp130   | Kotha: Logistic Regression                        |  400 |     0.9075 |   0.8775 |    0.935  |
|  47 | hosp130   | Sodar: Amplitude-QSVM (MinMax, paper)             |  600 |     0.9117 |   0.8867 |    0.935  |
|  48 | hosp130   | Sodar: Amplitude-QSVM (standardised)              |  600 |     0.9117 |   0.8883 |    0.9317 |
|  49 | hosp130   | Sodar: Angle-QSVM                                 |  600 |     0.9117 |   0.8883 |    0.9333 |
|  50 | hosp130   | Sodar: Logistic Regression                        |  600 |     0.9067 |   0.8833 |    0.9283 |
|  51 | hosp130   | Sodar: Majority class                             |  600 |     0.9117 |   0.8883 |    0.9333 |
|  52 | hosp130   | Sodar: RBF-SVM                                    |  600 |     0.9117 |   0.8883 |    0.9333 |
|  53 | hosp130   | Upama: Angle-QSVM (DML top-10)                    |  400 |     0.9125 |   0.8825 |    0.94   |
|  54 | hosp130   | Upama: Linear SVM (DML top-10)                    |  400 |     0.9125 |   0.885  |    0.94   |
|  55 | hosp130   | Upama: RF (DML top-10)                            |  400 |     0.9125 |   0.885  |    0.94   |
|  56 | hosp130   | Upama: RF (all features)                          |  400 |     0.9125 |   0.885  |    0.94   |
|  57 | hosp130   | Yadav: RBF-SVM                                    |  400 |     0.9125 |   0.885  |    0.94   |
|  58 | hosp130   | Yadav: Z-QSVM bw=1                                |  400 |     0.9125 |   0.885  |    0.9375 |
|  59 | hosp130   | Yadav: ZZ-QSVM (full) bw=1                        |  400 |     0.9125 |   0.885  |    0.94   |
|  60 | hosp130   | Yadav: ZZ-QSVM (linear) bw=1                      |  400 |     0.9125 |   0.8825 |    0.94   |
|  61 | hosp130   | Young: QISICGM ensemble (+GMM, global)            | 7000 |     0.972  |   0.968  |    0.976  |
|  62 | hosp130   | Zhang: LightGBM                                   | 2000 |     0.9105 |   0.899  |    0.9235 |
|  63 | hosp130   | Zhang: LightGBM + GIF weights                     | 2000 |     0.9035 |   0.8905 |    0.9165 |
|  64 | pima      | Daga: Amplitude-QSVM                              |  768 |     0.638  |   0.6029 |    0.6706 |
|  65 | pima      | Daga: Angle-QSVM                                  |  768 |     0.7578 |   0.7279 |    0.7891 |
|  66 | pima      | Daga: Hard-Voting (4 kernels)                     |  768 |     0.6667 |   0.6328 |    0.6992 |
|  67 | pima      | Daga: Logistic Regression                         |  768 |     0.7747 |   0.7448 |    0.8047 |
|  68 | pima      | Daga: Majority class                              |  768 |     0.651  |   0.6159 |    0.6849 |
|  69 | pima      | Daga: OOF-Stacking (LR meta)                      |  768 |     0.7591 |   0.7305 |    0.7891 |
|  70 | pima      | Daga: RBF-SVM                                     |  768 |     0.7604 |   0.7305 |    0.7904 |
|  71 | pima      | Daga: Random Forest                               |  768 |     0.7669 |   0.7383 |    0.7969 |
|  72 | pima      | Daga: ZZ-QSVM                                     |  768 |     0.6628 |   0.6302 |    0.6966 |
|  73 | pima      | Kiran: Angle-QSVM (same 4 PCs)                    |  154 |     0.7143 |   0.6429 |    0.7857 |
|  74 | pima      | Kiran: HQCAN VQC (4q, 2 layers)                   |  154 |     0.6753 |   0.6037 |    0.7468 |
|  75 | pima      | Kiran: Logistic Regression (same 4 PCs, balanced) |  154 |     0.6948 |   0.6299 |    0.7662 |
|  76 | pima      | Kiran: RBF-SVM (same 4 PCs, balanced)             |  154 |     0.6818 |   0.6104 |    0.7532 |
|  77 | pima      | Kotha: DiaBeats RF                                |  154 |     0.7597 |   0.6946 |    0.8247 |
|  78 | pima      | Kotha: Logistic Regression                        |  154 |     0.7143 |   0.6429 |    0.7857 |
|  79 | pima      | Sodar: Amplitude-QSVM (MinMax, paper)             |  231 |     0.6407 |   0.5801 |    0.7056 |
|  80 | pima      | Sodar: Amplitude-QSVM (standardised)              |  231 |     0.6667 |   0.6061 |    0.7273 |
|  81 | pima      | Sodar: Angle-QSVM                                 |  231 |     0.7446 |   0.6883 |    0.8009 |
|  82 | pima      | Sodar: Logistic Regression                        |  231 |     0.7446 |   0.6883 |    0.8009 |
|  83 | pima      | Sodar: Majority class                             |  231 |     0.6494 |   0.5887 |    0.7056 |
|  84 | pima      | Sodar: RBF-SVM                                    |  231 |     0.7489 |   0.6883 |    0.8052 |
|  85 | pima      | Upama: Angle-QSVM (DML top-7)                     |  154 |     0.7532 |   0.6818 |    0.8182 |
|  86 | pima      | Upama: Linear SVM (DML top-7)                     |  154 |     0.7208 |   0.6558 |    0.7922 |
|  87 | pima      | Upama: RF (DML top-7)                             |  154 |     0.7792 |   0.7143 |    0.8377 |
|  88 | pima      | Upama: RF (all features)                          |  154 |     0.7597 |   0.6883 |    0.8247 |
|  89 | pima      | Yadav: RBF-SVM                                    |  154 |     0.7468 |   0.6818 |    0.8117 |
|  90 | pima      | Yadav: Z-QSVM bw=1                                |  154 |     0.7338 |   0.6623 |    0.8052 |
|  91 | pima      | Yadav: ZZ-QSVM (full) bw=1                        |  154 |     0.6494 |   0.5714 |    0.7273 |
|  92 | pima      | Yadav: ZZ-QSVM (linear) bw=1                      |  154 |     0.6558 |   0.5844 |    0.7338 |
|  93 | pima      | Young: QISICGM ensemble (+GMM, global)            | 2768 |     0.8974 |   0.8851 |    0.9082 |
|  94 | pima      | Zhang: LightGBM                                   |  768 |     0.7513 |   0.7214 |    0.7826 |
|  95 | pima      | Zhang: LightGBM + GIF weights                     |  768 |     0.7682 |   0.737  |    0.7982 |
|  96 | sylhet    | Daga: Amplitude-QSVM                              |  520 |     0.9192 |   0.8942 |    0.9423 |
|  97 | sylhet    | Daga: Angle-QSVM                                  |  520 |     0.9481 |   0.9288 |    0.9673 |
|  98 | sylhet    | Daga: Hard-Voting (4 kernels)                     |  520 |     0.9673 |   0.9519 |    0.9808 |
|  99 | sylhet    | Daga: Logistic Regression                         |  520 |     0.9288 |   0.9058 |    0.95   |
| 100 | sylhet    | Daga: Majority class                              |  520 |     0.6154 |   0.5731 |    0.6558 |
| 101 | sylhet    | Daga: OOF-Stacking (LR meta)                      |  520 |     0.9635 |   0.9462 |    0.9788 |
| 102 | sylhet    | Daga: RBF-SVM                                     |  520 |     0.9731 |   0.9577 |    0.9865 |
| 103 | sylhet    | Daga: Random Forest                               |  520 |     0.9808 |   0.9673 |    0.9923 |
| 104 | sylhet    | Daga: ZZ-QSVM                                     |  520 |     0.9327 |   0.9096 |    0.9538 |
| 105 | sylhet    | Kiran: Angle-QSVM (same 4 PCs)                    |  104 |     0.9519 |   0.9038 |    0.9904 |
| 106 | sylhet    | Kiran: HQCAN VQC (4q, 2 layers)                   |  104 |     0.7885 |   0.7019 |    0.8654 |
| 107 | sylhet    | Kiran: Logistic Regression (same 4 PCs, balanced) |  104 |     0.9231 |   0.8654 |    0.9712 |
| 108 | sylhet    | Kiran: RBF-SVM (same 4 PCs, balanced)             |  104 |     0.9327 |   0.8846 |    0.9808 |
| 109 | sylhet    | Kotha: DiaBeats RF                                |  104 |     0.9904 |   0.9712 |    1      |
| 110 | sylhet    | Kotha: Logistic Regression                        |  104 |     0.9423 |   0.8942 |    0.9808 |
| 111 | sylhet    | Sodar: Amplitude-QSVM (MinMax, paper)             |  156 |     0.9487 |   0.9103 |    0.9808 |
| 112 | sylhet    | Sodar: Amplitude-QSVM (standardised)              |  156 |     0.9103 |   0.8652 |    0.9487 |
| 113 | sylhet    | Sodar: Angle-QSVM                                 |  156 |     0.9423 |   0.9038 |    0.9744 |
| 114 | sylhet    | Sodar: Logistic Regression                        |  156 |     0.9231 |   0.8782 |    0.9615 |
| 115 | sylhet    | Sodar: Majority class                             |  156 |     0.6154 |   0.5385 |    0.6859 |
| 116 | sylhet    | Sodar: RBF-SVM                                    |  156 |     0.9615 |   0.9295 |    0.9872 |
| 117 | sylhet    | Upama: Angle-QSVM (DML top-10)                    |  104 |     0.9423 |   0.8942 |    0.9808 |
| 118 | sylhet    | Upama: Linear SVM (DML top-10)                    |  104 |     0.9423 |   0.8942 |    0.9808 |
| 119 | sylhet    | Upama: RF (DML top-10)                            |  104 |     0.9808 |   0.9519 |    1      |
| 120 | sylhet    | Upama: RF (all features)                          |  104 |     0.9904 |   0.9712 |    1      |
| 121 | sylhet    | Yadav: RBF-SVM                                    |  104 |     0.9808 |   0.9519 |    1      |
| 122 | sylhet    | Yadav: Z-QSVM bw=1                                |  104 |     0.9808 |   0.9519 |    1      |
| 123 | sylhet    | Yadav: ZZ-QSVM (full) bw=1                        |  104 |     0.9423 |   0.8942 |    0.9808 |
| 124 | sylhet    | Yadav: ZZ-QSVM (linear) bw=1                      |  104 |     0.9519 |   0.9038 |    0.9904 |
| 125 | sylhet    | Young: QISICGM ensemble (+GMM, global)            | 1874 |     0.9957 |   0.9925 |    0.9984 |
| 126 | sylhet    | Zhang: LightGBM                                   |  520 |     0.9712 |   0.9558 |    0.9846 |
| 127 | sylhet    | Zhang: LightGBM + GIF weights                     |  520 |     0.9615 |   0.9442 |    0.9769 |
