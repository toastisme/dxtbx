from __future__ import absolute_import, division, print_function

import pickle

from dxtbx import IncorrectFormatError
from dxtbx.format.FormatFile import FormatFile as Format


def test_pickle_incorrect_format():
    ex = IncorrectFormatError(Format, "Some message")
    pickle.loads(pickle.dumps(ex))
