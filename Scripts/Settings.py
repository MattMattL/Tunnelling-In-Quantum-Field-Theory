class Settings:
	""" A class containing calculation settings and formatting options. The
		objects shown below are initialised to the default options.

		To change settings, copy the objects into main() and re-assign new
		settings.
	"""

	# calculation settings
	EPSILON = 0.2 # works in the range [0.094, 0.380]
	NUM_RHOS = 1000
	NUM_EPSILONS = 10

	# plot settings
	GRAPH_COLOUR = 'black'

	ENABLE_ANALYTIC_PLOT = True
	ENABLE_NUMERICAL_PLOT = True

	ANALYTIC_EPSILON_RANGE = [0.05, 0.38]

	LINE_WIDTH = 1
	DOTTED_LINE_WIDTH = 0.5
	MARKER_SIZE = 3

	LINE_STYLE = '-'
	DOTTED_LINE_STYLE = ':'
	MARKER_STYLE = 'o'

	# other settings
	ENABLE_DEBUGGING = False