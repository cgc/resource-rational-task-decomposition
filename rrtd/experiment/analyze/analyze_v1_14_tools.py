import json
import numpy as np
from analyze_v1_12_tools import ExperimentData, ParticipantExperimentData
import rrtd

class ParticipantExperimentData(ParticipantExperimentData):
    @property
    def surveydata(self):
        rows = list(self.filtered_rows(lambda d: d['trial_type'] == 'survey-multi-choice'))
        assert len(rows) == 1
        row, = rows
        responses = json.loads(row.data['responses'])
        assert list(responses.keys()) == ['Q0']
        return dict(picture_draw=responses['Q0'])

class ExperimentData(ExperimentData):
    def __init__(self, *args, **kwargs):
        p_cls = kwargs.pop('ParticipantExperimentDataCls', ParticipantExperimentData)
        super().__init__(*args, ParticipantExperimentDataCls=p_cls, **kwargs)

    def nodraw_filter(self, *, look_never=True, look_rarely=False):
        assert look_never ^ look_rarely, 'must have never or rarely active, but not both'
        if look_never:
            valid = ('Did not draw/take picture',)
        elif look_rarely:
            valid = ('Did not draw/take picture', 'Rarely looked')
        return {
            p.pid
            for p in self.participants
            if p.questiondata['picture_draw'] in valid
        }
