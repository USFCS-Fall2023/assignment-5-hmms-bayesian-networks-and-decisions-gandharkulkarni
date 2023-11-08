import HMM as driver
from HMM import HMM

def q2_submission():
    hmm_obj = HMM()
    filename = 'partofspeech.browntags.trained'
    print('-------Generate method-------')
    hmm_obj.load(filename)
    print('Token: 20')
    print(hmm_obj.generate(20))

    print('-------Forward Algorithm-------')
    hmm_obj = HMM()
    hmm_obj.load(filename)
    observations = driver.read_obs_file('ambiguous_sents.obs')
    for observation in observations:
        output = hmm_obj.forward(observation)
        print(output)
        print(observation.outputseq)

    print('-------Viterbi Algorithm-------')
    hmm_obj = HMM()
    hmm_obj.load(filename)
    observations = driver.read_obs_file('ambiguous_sents.obs')
    for observation in observations:
        output = hmm_obj.viterbi(observation)
        print(output)
        print(observation.outputseq)


print('\n\n Question 2 \n\n')
q2_submission()

print('\n\n Question 3 \n\n')
import alarm
import carnet

