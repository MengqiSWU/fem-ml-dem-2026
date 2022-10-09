from FEMxEPxML.constitutive import constitutiveSingle, ConstitutiveMask

"""
    This file is used to develope the nonlinear elastic model
     
     e. g. Duncan EB model.
"""


class nonlinearCons(constitutiveSingle):
    def __init__(self, p0, ndim, explicit_flag):
        constitutiveSingle.__init__(self, p0, ndim)
        self.explcit_flag = explicit_flag

    def get_D(self):
        pass