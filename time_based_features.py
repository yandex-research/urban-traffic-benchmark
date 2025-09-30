import numpy as np
import pandas as pd


def auto_add_time_based_features_periods(time_based_features_periods, train_timespan, freq):
    if (
        train_timespan >= pd.to_timedelta('5 days') and
        freq < pd.to_timedelta('1 day') and
        'time-in-day' not in time_based_features_periods
    ):
        time_based_features_periods.append('time-in-day')

    if train_timespan >= pd.to_timedelta('14 days') and 'day-in-week' not in time_based_features_periods:
        time_based_features_periods.append('day-in-week')

    if train_timespan >= pd.to_timedelta('730 days') and 'week-in-year' not in time_based_features_periods:
        time_based_features_periods.append('week-in-year')

    if train_timespan >= pd.to_timedelta('730 days') and 'month-in-year' not in time_based_features_periods:
        time_based_features_periods.append('month-in-year')


def get_integer_encodings_for_values(values):
    unique_values = set(values)
    value_to_int = {value: i for i, value in enumerate(sorted(unique_values))}
    integer_encodings = np.array([value_to_int[value] for value in values])

    return integer_encodings


def get_time_in_hour_integer_feature_encodings(timestamps):
    values = list(zip(timestamps.minute, timestamps.second))
    integer_feature_encodings = get_integer_encodings_for_values(values)

    return integer_feature_encodings


def get_time_in_day_integer_feature_encodings(timestamps):
    values = list(zip(timestamps.hour, timestamps.minute, timestamps.second))
    integer_feature_encodings = get_integer_encodings_for_values(values)

    return integer_feature_encodings


def get_hour_in_day_integer_feature_encodings(timestamps):
    return timestamps.hour.values


def get_day_in_week_integer_feature_encodings(timestamps):
    return timestamps.day_of_week.values


def get_day_in_month_integer_feature_encodings(timestamps):
    return timestamps.day.values - 1


def get_day_in_year_integer_feature_encodings(timestamps):
    return timestamps.day_of_year.values - 1


def get_week_in_year_integer_feature_encodings(timestamps):
    return (timestamps.day_of_year.values - 1) // 7


def get_month_in_year_integer_feature_encodings(timestamps):
    return timestamps.month.values - 1


GET_INTEGER_FEATURE_ENCODINGS_FUNC_DICT = {
    'time-in-hour': get_time_in_hour_integer_feature_encodings,
    'time-in-day': get_time_in_day_integer_feature_encodings,
    'hour-in-day': get_hour_in_day_integer_feature_encodings,
    'day-in-week': get_day_in_week_integer_feature_encodings,
    'day-in-month': get_day_in_month_integer_feature_encodings,
    'day-in-year': get_day_in_year_integer_feature_encodings,
    'week-in-year': get_week_in_year_integer_feature_encodings,
    'month-in-year': get_month_in_year_integer_feature_encodings
}


def get_integer_feature_encodings(timestamps, period):
    return GET_INTEGER_FEATURE_ENCODINGS_FUNC_DICT[period](timestamps)


def get_time_in_hour_period_lengths(timestamps, integer_feature_encodings):
    return np.full(len(timestamps), fill_value=integer_feature_encodings.max() + 1)


def get_time_in_day_period_lengths(timestamps, integer_feature_encodings):
    return np.full(len(timestamps), fill_value=integer_feature_encodings.max() + 1)


def get_hour_in_day_period_lengths(timestamps, integer_feature_encodings):
    return np.full(len(timestamps), fill_value=24)


def get_day_in_week_period_lengths(timestamps, integer_feature_encodings):
    return np.full(len(timestamps), fill_value=7)


def get_day_in_month_period_lengths(timestamps, integer_feature_encodings):
    return timestamps.days_in_month.values


def get_day_in_year_period_lengths(timestamps, integer_feature_encodings):
    return 365 + timestamps.is_leap_year.values


def get_week_in_year_period_lengths(timestamps, integer_feature_encodings):
    return np.full(len(timestamps), fill_value=52)


def get_month_in_year_period_lengths(timestamps, integer_feature_encodings):
    return np.full(len(timestamps), fill_value=12)


GET_PERIOD_LENGTHS_FUNC_DICT = {
    'time-in-hour': get_time_in_hour_period_lengths,
    'time-in-day': get_time_in_day_period_lengths,
    'hour-in-day': get_hour_in_day_period_lengths,
    'day-in-week': get_day_in_week_period_lengths,
    'day-in-month': get_day_in_month_period_lengths,
    'day-in-year': get_day_in_year_period_lengths,
    'week-in-year': get_week_in_year_period_lengths,
    'month-in-year': get_month_in_year_period_lengths
}


def get_period_lengths(timestamps, integer_feature_encodings, period):
    return GET_PERIOD_LENGTHS_FUNC_DICT[period](timestamps, integer_feature_encodings)


def get_one_hot_time_based_features(integer_feature_encodings, period):
    num_categories = integer_feature_encodings.max() + 1
    one_hot_time_based_features = np.zeros((len(integer_feature_encodings), num_categories), dtype=np.float32)
    one_hot_time_based_features[np.arange(len(integer_feature_encodings)), integer_feature_encodings] = 1

    one_hot_time_based_feature_names = [period.replace('-', '_') + f'_{i}' for i in range(num_categories)]

    return one_hot_time_based_features, one_hot_time_based_feature_names


def get_sin_cos_time_based_features(integer_feature_encodings, period_lengths, period):
    x = integer_feature_encodings / period_lengths * (np.pi * 2)
    sin_cos_time_based_features = np.stack([np.sin(x), np.cos(x)], axis=1, dtype=np.float32)

    sin_cos_time_based_feature_names = [period.replace('-', '_') + '_sin', period.replace('-', '_') + '_cos']

    return sin_cos_time_based_features, sin_cos_time_based_feature_names


def create_time_based_features(unix_timestamps, time_based_features_types, time_based_features_periods,
                               last_train_timestamp_idx):
    if not time_based_features_types or not time_based_features_periods:
        return (
            np.empty((len(unix_timestamps), 1, 0), dtype=np.float32), [],
            np.empty((len(unix_timestamps), 1, 0), dtype=np.float32), []
        )

    timestamps = pd.to_datetime(unix_timestamps, origin='unix', unit='s')
    timestamps = pd.DatetimeIndex(timestamps, freq='infer')

    if 'auto' in time_based_features_periods:
        train_timespan = timestamps[last_train_timestamp_idx] - timestamps[0] + timestamps.freq
        time_based_features_periods = list(time_based_features_periods)
        time_based_features_periods.remove('auto')
        auto_add_time_based_features_periods(time_based_features_periods=time_based_features_periods,
                                             train_timespan=train_timespan,
                                             freq=timestamps.freq)

    time_based_features, time_based_feature_names = [], []
    for period in time_based_features_periods:
        integer_feature_encodings = get_integer_feature_encodings(timestamps=timestamps, period=period)

        if 'one-hot' in time_based_features_types:
            cur_period_one_hot_time_based_features, cur_period_one_hot_time_based_feature_names = (
                get_one_hot_time_based_features(integer_feature_encodings=integer_feature_encodings,
                                                period=period)
            )

            time_based_features.append(cur_period_one_hot_time_based_features)
            time_based_feature_names += cur_period_one_hot_time_based_feature_names

        if 'sin-cos' in time_based_features_types:
            period_lengths = get_period_lengths(
                timestamps=timestamps, integer_feature_encodings=integer_feature_encodings, period=period
            )
            cur_period_sin_cos_time_based_features, cur_period_sin_cos_time_based_feature_names = (
                get_sin_cos_time_based_features(integer_feature_encodings=integer_feature_encodings,
                                                period_lengths=period_lengths,
                                                period=period)
            )

            time_based_features.append(cur_period_sin_cos_time_based_features)
            time_based_feature_names += cur_period_sin_cos_time_based_feature_names

    time_based_features = np.concatenate(time_based_features, axis=1)

    time_based_features = np.expand_dims(time_based_features, axis=1)

    numerical_time_based_features_mask = np.zeros(time_based_features.shape[2], dtype=bool)

    return time_based_features, time_based_feature_names, numerical_time_based_features_mask
