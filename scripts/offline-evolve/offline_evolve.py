# Offline evolution scheme
# - usage: python start.py @folder/settings-file.conf
# - We use a population of constant size population_size
# - Each robot is evaluated for evaluation_time seconds
# - The average speed during this evaluation is the fitness (almost)
# - We do parent selection using a binary tournament: select two parents at
#   random, the one with the best fitness is parent 1, repeat for parent 2.
# - Using this mechanism, we generate num_children children
# - After evaluation of the children, we either do:
# -- Plus scheme, sort *all* robots by fitness
# -- Comma scheme, get rid of the parents and continue with children only
from __future__ import absolute_import
import sys
import time
import os
import shutil
import random
import csv
import itertools
import logging
import trollius
from trollius import From, Return
from pprint import pprint

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../../')

from sdfbuilder import Pose
from sdfbuilder.math import Vector3

from revolve.util import wait_for

from tol.manage.robot import Robot
from tol.config import parser
from tol.manage import World
from tol.logging import logger, output_console
from tol.util.analyze import list_extremities, count_joints, count_motors, count_extremities, count_connections

# Log output to console
#output_console()
logger.setLevel(logging.INFO)

# Add offline evolve arguments
parser.add_argument(
    '--population-size',
    default=10, type=int,
    help="Population size in each generation."
)

parser.add_argument(
    '--num-children',
    default=10, type=int,
    help="The number of children produced in each generation."
)


def str2bool(v):
    return v.lower() == "true" or v == "1"

def printnow(s):
    print(s)
    sys.stdout.flush()

parser.add_argument(
    '--keep-parents',
    default=True, type=str2bool,
    help="Whether or not to discard the parents after each generation. This determines the strategy, + or ,."
)

parser.add_argument(
    '--num-generations',
    default=200, type=int,
    help="The number of generations to simulate."
)

parser.add_argument(
    '--disable-evolution',
    default=False, type=str2bool,
    help="Useful as a baseline test - if set to true, new robots are generated"
         " every time rather than evolving them."
)

parser.add_argument(
    '--disable-selection',
    default=False, type=str2bool,
    help="Useful as a different baseline test - if set to true, robots reproduce,"
         " but parents are selected completely random."
)

parser.add_argument(
    '--disable-fitness',
    default=False, type=str2bool,
    help="Another baseline testing option, sorts robots randomly rather "
         "than selecting the top pairs. This only matters if parents are kept."
)

parser.add_argument(
    '--evaluation-threshold',
    default=10.0, type=float,
    help="Maximum number of seconds one evaluation can take before the "
         "decision is made to restart from snapshot. The assumption is "
         "that the world may have become slow and restarting will help."
)

parser.add_argument(
    '--restart-interval',
    default=10, type=int,
    help="Number of generations to run before restarting from snapshot."
         "The assumption is that restarting will avoid memory leak crashes and slow simulation."
)

parser.add_argument(
    '--world-file',
    default='offline-evolve.world', type=str,
    help="The world file to evolve in."
)

parser.add_argument(
    '--num-evolutions',
    default=10, type=int,
    help="The number of times to repeat the experiment (runs)."
)

# parser.add_argument(
#     '--disable-',
#     default=10, type=int,
#     help="The number of times to repeat the experiment (runs)."
# )


# parser.add_argument(
#     '--robot-id-base',
#     default=0, type=int,
#     help="Robot ID to start from. To avoid name clahes when doing runs on multiple machines and combining later."
# )

parser.add_argument(
    '--start-run',
    default=0, type=int,
    help="Run to start from. Useful when doing runs on multiple machines."
)

parser.add_argument(
    '--robot-id-stride',
    default=100000, type=int,
    help="stride mutiplied by run gives the robot id to start from. To avoid ID clashes when running on multiple machines. Should be larger than the expected number of generated robots in a run"
)


class OfflineEvoManager(World):
    """
    Extended world manager for the offline evolution script
    """

    def __init__(self, conf, _private):
        """

        :param conf:
        :param _private:
        :return:
        """
        super(OfflineEvoManager, self).__init__(conf, _private)

        self._snapshot_data = {}
        # Output files
        csvs = {
            'generations': ['run', 'gen', 'robot_id', 'vel', 'dvel', 'fitness', 't_eval'],
            'robot_details': ['robot_id', 'extremity_id', 'extremity_size', 'joint_count', 'motor_count'],
            'generation_details': ['run', 'gen', 'gen_time']
        }
        self.csv_files = {k: {'filename': None, 'file': None, 'csv': None,
                              'header': csvs[k]}
                          for k in csvs}

        self.current_run = 0

        if self.output_directory:
            for k in self.csv_files:
                fname = os.path.join(self.output_directory, k + '.csv')
                self.csv_files[k]['filename'] = fname
                if self.do_restore:
                    shutil.copy(fname + '.snapshot', fname)
                    f = open(fname, 'ab', buffering=1)
                else:
                    f = open(fname, 'wb', buffering=1)

                self.csv_files[k]['file'] = f
                self.csv_files[k]['csv'] = csv.writer(f, delimiter=',')

                if not self.do_restore:
                    self.csv_files[k]['csv'].writerow(self.csv_files[k]['header'])

    def robots_header(self):
        return Robot.header()

    @classmethod
    @trollius.coroutine
    def create(cls, conf):
        """
        Coroutine to instantiate a Revolve.Angle WorldManager
        :param conf:
        :return:
        """
        self = cls(_private=cls._PRIVATE, conf=conf)
        yield From(self._init())
        raise Return(self)

    @trollius.coroutine
    def create_snapshot(self):
        """
        Copy the generations file in the snapshot
        :return:
        """
        ret = yield From(super(OfflineEvoManager, self).create_snapshot())
        if not ret:
            raise Return(ret)

        for k in self.csv_files:
            entry = self.csv_files[k]
            if entry['file']:
                entry['file'].flush()
                shutil.copy(entry['filename'], entry['filename'] + '.snapshot')

    @trollius.coroutine
    def get_snapshot_data(self):
        """
        :return:
        """
	#print("Tree", tree)
        #pprint(vars(tree))
        #print("Bbox", bbox)
        data = yield From(super(OfflineEvoManager, self).get_snapshot_data())
        data.update(self._snapshot_data)
        raise Return(data)

    @trollius.coroutine
    def evaluate_pair(self, tree, bbox, parents=None):
        """
        Evaluates a single robot tree.
        :param tree:
        :param bbox:
        :param parents:
        :return: Evaluated Robot object
        """
        # Pause the world just in case it wasn't already
        yield From(wait_for(self.pause(True)))
       # print("Tree", tree)
        #pprint(vars(tree))
        #print("Bbox", bbox)
        pose = Pose(position=Vector3(0, 0, -bbox.min.z))
        fut = yield From(self.insert_robot(tree, pose, parents=parents))
        robot = yield From(fut)

        max_age = self.conf.evaluation_time + self.conf.warmup_time

        # Unpause the world to start evaluation
        yield From(wait_for(self.pause(False)))

        before = time.time()

        while True:
            if robot.age() >= max_age:
                break

            # Sleep for the pose update frequency, which is about when
            # we expect a new age update.
            yield From(trollius.sleep(1.0 / self.state_update_frequency))

        yield From(wait_for(self.delete_robot(robot)))
        yield From(wait_for(self.pause(True)))

        diff = time.time() - before
        if diff > self.conf.evaluation_threshold:
            sys.stderr.write("Evaluation threshold exceeded, shutting down with nonzero status code.\n")
            sys.stderr.flush()
            sys.exit(15)

        raise Return(robot)

    @trollius.coroutine
    def evaluate_population(self, trees, bboxes, parents=None):
        """
        :param trees:
        :param bboxes:
        :param parents:
        :return:
        """
        if parents is None:
            parents = [None for _ in trees]

        pairs = []
        printnow("--- Evaluating population ---")
        start_time = time.time()
        for tree, bbox, par in itertools.izip(trees, bboxes, parents):
            printnow("Evaluating individual...")
            sys.stdout.flush()

            before = time.time()
            robot = yield From(self.evaluate_pair(tree, bbox, par))
            pairs.append((robot, time.time() - before))
            print("Done in %.2f s." % (time.time() - before) )
            sys.stdout.flush()

        diff = time.time() - start_time
        print("--- Done evaluating population. ---")
        print("Population evaluation time: %.2f s. " % diff)

        raise Return(pairs)

    @trollius.coroutine
    def produce_generation(self, parents):
        """
        Produce the next generation of robots from
        the current.
        :param parents:
        :return:
        """
        printnow("--- Producing generation ---")
        trees = []
        bboxes = []
        parent_pairs = []

        while len(trees) < self.conf.num_children:
            printnow("Producing individual...")
            if self.conf.disable_selection:
                p1, p2 = random.sample(parents, 2)
            else:
		p1, p2 = yield From(select_parents(parents, self.conf)) 

            for j in xrange(self.conf.max_mating_attempts):
                pair = yield From(self.attempt_mate(p1, p2))
                if pair:
                    trees.append(pair[0])
                    bboxes.append(pair[1])
                    parent_pairs.append((p1, p2))
                    break

            print("Done.")

        print("--- Done producing generation. ---")
        raise Return(trees, bboxes, parent_pairs)

    @trollius.coroutine
    def helper(self, tree):
        #print("in helper")
        ret = yield From(self.analyze_tree(tree))
        if ret is None:
             # Error already shown
             print("ERROR ret is none!")
             #continue

        coll, bbox, robot = ret
        #print(tree)
        #print("BBOX")
        #print(bbox)
	#raise Return(bbox)
        if bbox:
            raise Return(bbox)

        logger.error("Error in helper")
        raise Return(None)

    @trollius.coroutine
    def log_generation(self, evo, generation, pairs, generation_eval_time=0):
        """
        :param evo: The evolution run
        :param generation:
        :param pairs: List of tuples (robot, evaluation wallclock time)
        :return:
        """
        printnow("================== GENERATION %d DONE ==================" % generation)
        if not self.output_directory:
            return

        go = self.csv_files['generations']['csv']
        do = self.csv_files['robot_details']['csv']
	
        for robot, t_eval in pairs:
            #print("ROBOT analyze_treeeeeeeeeeeeeee")
	    ret = yield From(self.helper(robot.tree))
            #print(ret)
	    #print(list(ret))
	    #first, second = ret
	    #print(first)
	    #print(second)
            robot_id = robot.robot.id
            root = robot.tree.root
            #print("BBOX-fitnes from one robot", robot.fitness_bbox(ret))
            #print("Regular fitness", robot.fitness())
            go.writerow([evo, generation, robot.robot.id, robot.velocity(),
                         robot.displacement_velocity(), robot.fitness(), t_eval])

            # TODO Write this once when robot is written instead
            counter = 0
            for extr in list_extremities(root):
                num_joints = count_joints(extr)
                num_motors = count_motors(extr)
                do.writerow((robot_id, counter, len(extr), num_joints, num_motors))
                counter += 1


        gen_file = self.csv_files['generation_details']['csv']
        gen_file.writerow([evo,generation,generation_eval_time])


    @trollius.coroutine
    def run(self):
        """
        :return:
        """
        conf = self.conf

        if self.do_restore:
            print("Restoring from a previously cancelled / crashed experiment")
            # Recover from a previously cancelled / crashed experiment
            data = self.do_restore
            evo_start = data['evo_start']
            gen_start = data['gen_start']
            pairs = data['local_pairs']
        else:
            print ("Starting a fresh experiment")
            # Start at the specified run (default is 0)
            evo_start = conf.start_run
            self.robot_id = conf.robot_id_stride * conf.start_run
            gen_start = 1
            pairs = None
        gen_count=0

        #the main run loop
        for evo in range(evo_start, conf.start_run + conf.num_evolutions):
            self.current_run = evo
            print("Current run: %d" % evo)

            if not pairs:
                # Only create initial population if we are not restoring from
                # a previous experiment.
                before = time.time()
                printnow("Generating initial population...")
                trees, bboxes = yield From(self.generate_population(conf.population_size))
                printnow("Evaluating initial population...")
                pairs = yield From(self.evaluate_population(trees, bboxes))
                printnow("Done evaluating initial population...")
                diff = time.time() - before
                yield From(self.log_generation(evo, 0, pairs, diff))
		printnow("done with logging!")
                gen_count += 1

            for generation in xrange(gen_start, conf.num_generations):
                # snapshot data every generation, overhead is not too big
                self._snapshot_data = {
                    "local_pairs": pairs,
                    "gen_start": generation,
                    "evo_start": evo
                }
                yield From(self.create_snapshot())
                print("Created snapshot of experiment state")

                
		# restart every n generations to avoid crash from memory leaks or similar
                #print("gencount", gen_count)
                #print("self.conf.restart_interval", self.conf.restart_interval)
		if gen_count==self.conf.restart_interval: #make this a parameter
                    print("Initiating scheduled shutdown and restart from snapshot...")
                    sys.exit(22)
                gen_count += 1

                before = time.time()

                # Produce the next generation and evaluate them
                robots = [p[0] for p in pairs]
                if conf.disable_evolution:
                    child_trees, child_bboxes = yield From(
                        self.generate_population(conf.population_size))
                    parent_pairs = None
                else:
                    child_trees, child_bboxes, parent_pairs = yield From(self.produce_generation(robots))

                child_pairs = yield From(self.evaluate_population(child_trees, child_bboxes, parent_pairs))

                if conf.keep_parents:
                    pairs += child_pairs
                else:
                    pairs = child_pairs

                # Sort the bots and reduce to population size
                if conf.disable_fitness:
                    random.shuffle(pairs)
                else:
                    # print("in else")
                    #print [bbox for bbox in child_bboxes]
		    #Her har vi tilgang til pairs som sikkert innehar bbox! saa lag en metode i fitness til robot som taar inn den.
                    pairs = sorted(pairs, key=lambda r: r[0].fitness(), reverse=True)

                pairs = pairs[:conf.population_size]

                #display elapsed time of 1 generation
                diff = time.time() - before
                printnow("Generation time: %.2f s." % diff)

                yield From(self.log_generation(evo, generation, pairs, diff))


            # Clear "restore" parameters
            gen_start = 1
            pairs = None

        yield From(self.teardown())

    @trollius.coroutine
    def teardown(self):
        """
        :return:
        """
        yield From(super(OfflineEvoManager, self).teardown())
        for k in self.csv_files:
            if self.csv_files[k]['file']:
                self.csv_files[k]['file'].close()

@trollius.coroutine
def select_parent(parents, conf):
    """
    Select a parent using a binary tournament.
    :param parents:
    :param conf: Configuration object
    :return:
    """
    parents_random_sample = random.sample(parents, conf.tournament_size)
    ret = yield From(world.helper(parents_random_sample[0].tree))
    print("IN SELECT PAREEEEEEEEEEEEEEEEEEEEEEEEEEEEENT")
    #print(ret)
    #random_sample = random.sample(parents, conf.tournament_size);
    samples_with_bbox = []
    for index, robot in enumerate(parents_random_sample):
        bbox = yield From(world.helper(robot.tree))
        robot_with_bbox = {"bbox": bbox, "robot":robot}
        samples_with_bbox.append(robot_with_bbox)
    sorted_samples = sorted (samples_with_bbox, key=lambda o: o["robot"].fitness_bbox(o["bbox"]))
    print("SORTED SAMPLEEEEEEE", sorted_samples[-1])
    raise Return(sorted(random.sample(parents, conf.tournament_size), key=lambda r: r.fitness())[-1])

@trollius.coroutine
def select_parents(parents, conf):
    """
    :param parents:
    :param conf: Configuration object
    :return:
    """
    p1 = yield From(select_parent(parents, conf))
    p2 = yield From(select_parent(list(parent for parent in parents if parent != p1), conf))
    raise Return(p1, p2)

world = None
@trollius.coroutine
def run():
    """
    :return:
    """
    conf = parser.parse_args()

    global world
    world = yield From(OfflineEvoManager.create(conf))
    yield From(world.run())


def main():
    try:
        loop = trollius.get_event_loop()
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        print("Got Ctrl+C, shutting down.")


if __name__ == '__main__':
    main()
