from . import _GAR

# ---------------------------------------------------------------------------- #
# Average GAR class

class AverageGAR(_GAR):
  """ Simple synchronous average GAR class.
  """

  def __init__(self):
    pass

  def aggregate(self, gradients):
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    if len(gradients) > 1:
      return tf.add_n(gradients) / float(len(gradients))
    else:
      return gradients[0]
