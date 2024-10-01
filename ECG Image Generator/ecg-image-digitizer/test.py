import sys
import matlab.engine

positionOfPath = 1
sys.path.insert(positionOfPath, '.')

eng = matlab.engine.start_matlab()
eng.test_ecg_sequence_extraction(nargout=0)
eng.quit()
