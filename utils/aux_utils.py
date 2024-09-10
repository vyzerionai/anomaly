import abc
import pandas as pd
import numpy as np
from typing import Sequence
from scipy import signal


class BaseAuxiliaryFeature(metaclass= abc.ABCMeta):
  """Base class for augmenting raw feature with auxiliary features."""


  @abc.abstractmethod
  def append(self, flight_data: pd.DataFrame) -> pd.DataFrame:
    """Returns a new dataframe, appended with the auxiliary features."""

  @property
  @abc.abstractmethod
  def feature_names(self) -> Sequence[str]:
    """Returns the name of the features."""



class FourierAuxiliaryFeature(BaseAuxiliaryFeature):

  def __init__(self, selected_features: list[str], num_fourier_coeffs:int = 4):
    self._selected_features = selected_features
    self._num_fourier_coeffs = num_fourier_coeffs
    self._feature_names = None


  def append(self, flight_data: pd.DataFrame) -> pd.DataFrame:
    """Returns a new dataframe, appended with the auxiliary features."""

    (n_points, n_features) = flight_data.shape
    fnames = []


    for feature in self._selected_features:

      if feature.startswith('CHT'):
        sig = flight_data[feature] -  np.mean(flight_data[['CHT1', 'CHT2', 'CHT3', 'CHT4', 'CHT5', 'CHT6']], axis = 1)
      elif feature.startswith('EGT'):
        sig = flight_data[feature] -  np.mean(flight_data[['EGT1', 'EGT2', 'EGT3', 'EGT4', 'EGT5', 'EGT6']], axis = 1)
      else:
        sig = flight_data[feature]

      _, _, Sxx = self._compute_fourier_coeffs(sig)

      (n_fourier_features, n_fourier_points) = Sxx.shape
      diff_points = n_points - n_fourier_points
      points_prepend = int(diff_points/2.0)
      points_postpend = diff_points - points_prepend
      for f in range(self._num_fourier_coeffs):
        f_array = np.concatenate([np.zeros(points_prepend), Sxx[f, :],np.zeros(points_postpend)])
        fname = 'aux_fft%d_%s' %(f, feature)
        flight_data[fname] = f_array
        fnames.append(fname)
    if self._feature_names is None:
      self._feature_names = fnames
    return flight_data


  @property
  def feature_names(self) -> list[str]:
    return self._feature_names

  def _compute_fourier_coeffs(self, series: pd.Series) -> tuple[np.array, np.array, np.array]:
    fs = 1
    f, t, Sxx = signal.spectrogram(series, fs, nperseg = 256, window=('tukey', 0.5),
                                   scaling = 'spectrum', mode = 'magnitude', noverlap=255)

    return f, t, Sxx



def apply_auxiliary_features(flights: dict[tuple[str,str],
                                           tuple[str, pd.DataFrame]],
                             auxiliary_features: dict[str, BaseAuxiliaryFeature]):

  for flight_key in flights:
    for auxf in auxiliary_features:
      print('Adding auxiliary feature %s to flight %s' %(auxf, flight_key))
      flight_file = flights[flight_key][0]
      flight_data = flights[flight_key][1]
      if len(flights[flight_key]) == 4:
        initial_complaint = flights[flight_key][2]
        corrective_action = flights[flight_key][3]
      else:
        initial_complaint = None
        corrective_action = None
      flight_data = auxiliary_features[auxf].append(flight_data)
      flights[flight_key] = tuple([flight_file, flight_data, initial_complaint, corrective_action])

  return flights

