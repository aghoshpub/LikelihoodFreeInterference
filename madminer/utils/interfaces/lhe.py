from __future__ import absolute_import, division, print_function, unicode_literals

import six
import numpy as np
from collections import OrderedDict
import os
import logging

try:
    import xml.etree.cElementTree as ET

    use_celementtree = True
except ImportError:
    import xml.etree.ElementTree as ET

    use_celementtree = False

from madminer.utils.various import call_command, approx_equal, math_commands
from madminer.utils.particle import MadMinerParticle

logger = logging.getLogger(__name__)


def parse_lhe_file(
    filename,
    sampling_benchmark,
    observables,
    observables_required=None,
    observables_defaults=None,
    cuts=None,
    cuts_default_pass=None,
    efficiencies=None,
    efficiencies_default_pass=None,
    benchmark_names=None,
    is_background=False,
    energy_resolutions=None,
    pt_resolutions=None,
    eta_resolutions=None,
    phi_resolutions=None,
    k_factor=1.0,
    parse_events_as_xml=True,
):
    """ Extracts observables and weights from a LHE file """

    logger.debug("Parsing LHE file %s", filename)

    if parse_events_as_xml:
        logger.debug("Parsing header and events as XML with %sElementTree", "c" if use_celementtree else "")
    else:
        logger.debug(
            "Parsing header as XML with %sElementTree and events as text file", "c" if use_celementtree else ""
        )

    # Inputs
    if k_factor is None:
        k_factor = 1.0
    if observables_required is None:
        observables_required = {key: False for key in six.iterkeys(observables)}
    if observables_defaults is None:
        observables_defaults = {key: None for key in six.iterkeys(observables)}
    if is_background and benchmark_names is None:
        raise RuntimeError("Parsing background LHE files required benchmark names to be provided.")
    if cuts is None:
        cuts = OrderedDict()
    if cuts_default_pass is None:
        cuts_default_pass = {key: False for key in six.iterkeys(cuts)}
    if efficiencies is None:
        efficiencies = OrderedDict()
    if efficiencies_default_pass is None:
        efficiencies_default_pass = {key: 1.0 for key in six.iterkeys(efficiencies)}

    # Untar and open LHE file
    run_card = None
    for elem in _untar_and_parse_lhe_file(filename):
        if elem.tag == "MGRunCard":
            run_card = elem.text
            break
        else:
            continue

    # Figure out event weighting
    weight_norm_is_average = None
    n_events_runcard = None
    for line in run_card.splitlines():
        # Remove run card comments
        try:
            line, _ = line.split("!")
        except:
            pass

        # Separate in keys and values
        try:
            value, key = line.split("=")
        except:
            continue

        # Remove spaces
        value = value.strip()
        key = key.strip()

        # Parse entries
        if key == "nevents":
            n_events_runcard = float(value)
        if key == "event_norm":
            weight_norm_is_average = value == "average"

            logger.debug(
                "Found entry event_norm = %s in LHE header. Interpreting this as weight_norm_is_average " "= %s.",
                value,
                weight_norm_is_average,
            )

    if weight_norm_is_average is None:
        logger.warning(
            "Cannot read weight normalization mode (entry 'event_norm') from LHE file header. MadMiner "
            "will continue assuming that events are properly normalized. Please check this!"
        )

    # If necessary, rescale by number of events
    if weight_norm_is_average:
        if n_events_runcard is None:
            raise RuntimeError(
                "LHE weights have to be normalized, but MadMiner cannot read number of events (entry "
                "'nevents') from LHE file header."
            )

        k_factor = k_factor / n_events_runcard

    # Loop over events
    n_events_with_negative_weights = 0
    pass_cuts = [0 for _ in cuts]
    fail_cuts = [0 for _ in cuts]
    pass_efficiencies = [0 for _ in efficiencies]
    fail_efficiencies = [0 for _ in efficiencies]
    avg_efficiencies = [0 for _ in efficiencies]

    observations_all_events = []
    weights_all_events = []
    weight_names_all_events = None

    # Option one: XML parsing
    if parse_events_as_xml:
        events = _untar_and_parse_lhe_file(filename, ["event"])
        for i_event, event in enumerate(events):
            if (i_event + 1) % 100000 == 0:
                logger.info("  Processing event %d/%d", i_event + 1, n_events_runcard)

            # Parse event
            particles, weights, global_event_data = _parse_xml_event(event, sampling_benchmark)
            n_events_with_negative_weights, observations, pass_all, weight_names_all_events, weights = _parse_event(
                avg_efficiencies,
                cuts,
                cuts_default_pass,
                efficiencies,
                efficiencies_default_pass,
                energy_resolutions,
                eta_resolutions,
                fail_cuts,
                fail_efficiencies,
                n_events_with_negative_weights,
                observables,
                observables_defaults,
                observables_required,
                particles,
                pass_cuts,
                pass_efficiencies,
                phi_resolutions,
                pt_resolutions,
                weight_names_all_events,
                weights,
                global_event_data=global_event_data,
            )

            # Skip events that fail anything
            if not pass_all:
                continue

            # Store results
            observations_all_events.append(observations)
            weights_all_events.append(weights)

    # Option two: text parsing
    else:
        # Iterate over events in LHE file
        for i_event, (particles, weights) in enumerate(_parse_txt_events(filename, sampling_benchmark)):
            if (i_event + 1) % 100000 == 0:
                logger.info("  Processing event %d/%d", i_event + 1, n_events_runcard)

            n_events_with_negative_weights, observations, pass_all, weight_names_all_events, weights = _parse_event(
                avg_efficiencies,
                cuts,
                cuts_default_pass,
                efficiencies,
                efficiencies_default_pass,
                energy_resolutions,
                eta_resolutions,
                fail_cuts,
                fail_efficiencies,
                n_events_with_negative_weights,
                observables,
                observables_defaults,
                observables_required,
                particles,
                pass_cuts,
                pass_efficiencies,
                phi_resolutions,
                pt_resolutions,
                weight_names_all_events,
                weights,
            )

            # Skip events that fail anything
            if not pass_all:
                continue

            # Store results
            observations_all_events.append(observations)
            weights_all_events.append(weights)

    # Check results
    n_events_pass = _report_parse_results(
        avg_efficiencies,
        cuts,
        efficiencies,
        fail_cuts,
        fail_efficiencies,
        n_events_with_negative_weights,
        observations_all_events,
        pass_cuts,
        pass_efficiencies,
    )

    if n_events_pass == 0:
        logger.warning("  No observations remaining!")
        return None, None

    # Reformat observations to OrderedDicts with entries {name : (n_events,)}
    observations_all_events = list(map(list, zip(*observations_all_events)))  # transposes to (n_observables, n_events)
    observations_dict = OrderedDict()
    for key, values in zip(observables.keys(), observations_all_events):
        observations_dict[key] = np.asarray(values)

    # Reformat weightss and add k-factors to weights
    weights_all_events = np.array(weights_all_events)  # (n_events, n_weights)
    weights_all_events = k_factor * weights_all_events
    weights_all_events = OrderedDict(zip(weight_names_all_events, weights_all_events.T))

    # Background events
    if is_background:
        for benchmark_name in benchmark_names:
            weights_all_events[benchmark_name] = weights_all_events[sampling_benchmark]

    return observations_dict, weights_all_events


def _report_parse_results(
    avg_efficiencies,
    cuts,
    efficiencies,
    fail_cuts,
    fail_efficiencies,
    n_events_with_negative_weights,
    observations_all_events,
    pass_cuts,
    pass_efficiencies,
):
    for n_pass, n_fail, cut in zip(pass_cuts, fail_cuts, cuts):
        logger.debug("  %s / %s events pass cut %s", n_pass, n_pass + n_fail, cut)
    for n_pass, n_fail, efficiency in zip(pass_efficiencies, fail_efficiencies, efficiencies):
        logger.debug("  %s / %s events pass efficiency %s", n_pass, n_pass + n_fail, efficiency)
    for n_eff, efficiency, n_pass, n_fail in zip(avg_efficiencies, efficiencies, pass_efficiencies, fail_efficiencies):
        logger.debug("  average efficiency for %s is %s", efficiency, n_eff / (n_pass + n_fail))
    n_events_pass = len(observations_all_events)
    if len(cuts) > 0:
        logger.info("  %s events pass all cuts/efficiencies", n_events_pass)
    if n_events_with_negative_weights > 0:
        logger.warning("  %s events contain negative weights", n_events_with_negative_weights)
    return n_events_pass


def _parse_event(
    avg_efficiencies,
    cuts,
    cuts_default_pass,
    efficiencies,
    efficiencies_default_pass,
    energy_resolutions,
    eta_resolutions,
    fail_cuts,
    fail_efficiencies,
    n_events_with_negative_weights,
    observables,
    observables_defaults,
    observables_required,
    particles,
    pass_cuts,
    pass_efficiencies,
    phi_resolutions,
    pt_resolutions,
    weight_names_all_events,
    weights,
    global_event_data=None,
):
    # Negative weights?
    n_events_with_negative_weights = _report_negative_weights(n_events_with_negative_weights, weights)
    if weight_names_all_events is None:
        weight_names_all_events = list(weights.keys())
    weights = np.array(list(weights.values()))

    # Apply smearing
    particles_smeared = _smear_particles(
        particles, energy_resolutions, pt_resolutions, eta_resolutions, phi_resolutions
    )

    # Objects in event
    try:
        variables = _get_objects(
            particles_smeared, particles, pt_resolutions["met"], global_event_data=global_event_data
        )
    except (TypeError, IndexError):
        variables = _get_objects(particles_smeared, particles, met_resolution=None, global_event_data=global_event_data)

    # Observables
    observations, pass_all_observation = _parse_observations(
        observables, observables_defaults, observables_required, variables
    )

    # Cuts
    pass_all_cuts = True
    if pass_all_observation:
        pass_all_cuts = _parse_cuts(
            cuts, cuts_default_pass, fail_cuts, observables, observations, pass_all_cuts, pass_cuts, variables
        )

    # Efficiencies
    pass_all_efficiencies = True
    if pass_all_observation and pass_all_cuts:
        pass_all_efficiencies, total_efficiency = _parse_efficiencies(
            avg_efficiencies, efficiencies, efficiencies_default_pass, fail_efficiencies, pass_efficiencies, variables
        )

        if pass_all_efficiencies:
            weights *= total_efficiency

    pass_all = pass_all_cuts and pass_all_efficiencies and pass_all_observation
    return n_events_with_negative_weights, observations, pass_all, weight_names_all_events, weights


def _report_negative_weights(n_events_with_negative_weights, weights):
    n_negative_weights = np.sum(np.array(list(weights.values())) < 0.0)
    if n_negative_weights > 0:
        n_events_with_negative_weights += 1
        if n_events_with_negative_weights <= 3:
            logger.warning("Found %s negative weights in event. Weights: %s", n_negative_weights, weights)
        if n_events_with_negative_weights == 3:
            logger.warning("Skipping warnings about negative weights from now on...")
    return n_events_with_negative_weights


def _parse_observations(observables, observables_defaults, observables_required, variables):
    observations = []
    pass_all_observation = True
    for obs_name, obs_definition in six.iteritems(observables):
        if isinstance(obs_definition, six.string_types):
            try:
                observations.append(eval(obs_definition, variables))
            except (SyntaxError, NameError, TypeError, ZeroDivisionError, IndexError):
                if observables_required[obs_name]:
                    pass_all_observation = False

                default = observables_defaults[obs_name]
                if default is None:
                    default = np.nan
                observations.append(default)
        else:
            try:
                observations.append(
                    obs_definition(
                        variables["p_truth"], variables["l"], variables["a"], variables["j"], variables["met"]
                    )
                )
            except RuntimeError:
                if observables_required[obs_name]:
                    pass_all_observation = False

                default = observables_defaults[obs_name]
                if default is None:
                    default = np.nan
                observations.append(default)
    return observations, pass_all_observation


def _parse_efficiencies(
    avg_efficiencies, efficiencies, efficiencies_default_pass, fail_efficiencies, pass_efficiencies, variables
):
    # Apply efficiencies
    total_efficiency = 1.0
    pass_all_efficiencies = True
    for i_efficiency, (efficiency, default_pass) in enumerate(zip(efficiencies, efficiencies_default_pass)):
        try:
            efficiency_result = eval(efficiency, variables)
            if efficiency_result > 0.0:
                pass_efficiencies[i_efficiency] += 1
                total_efficiency *= efficiency_result
                avg_efficiencies[i_efficiency] += efficiency_result
            else:
                fail_efficiencies[i_efficiency] += 1
                pass_all_efficiencies = False

        except (SyntaxError, NameError, TypeError, ZeroDivisionError, IndexError):
            if default_pass > 0.0:
                pass_efficiencies[i_efficiency] += 1
                total_efficiency *= default_pass
                avg_efficiencies[i_efficiency] += default_pass
            else:
                fail_efficiencies[i_efficiency] += 1
                pass_all_efficiencies = False
    return pass_all_efficiencies, total_efficiency


def _parse_cuts(cuts, cuts_default_pass, fail_cuts, observables, observations, pass_all_cuts, pass_cuts, variables):
    # Objects for cuts
    for obs_name, obs_value in zip(observables.keys(), observations):
        variables[obs_name] = obs_value
    # Check cuts
    for i_cut, (cut, default_pass) in enumerate(zip(cuts, cuts_default_pass)):
        try:
            cut_result = eval(cut, variables)
            if cut_result:
                pass_cuts[i_cut] += 1
            else:
                fail_cuts[i_cut] += 1
                pass_all_cuts = False

        except (SyntaxError, NameError, TypeError, ZeroDivisionError, IndexError):
            if default_pass:
                pass_cuts[i_cut] += 1
            else:
                fail_cuts[i_cut] += 1
                pass_all_cuts = False
    return pass_all_cuts


def extract_nuisance_parameters_from_lhe_file(filename, systematics):
    """ Extracts the definition of nuisance parameters from the LHE file """

    logger.debug("Parsing nuisance parameter setup from LHE file at %s", filename)

    # Nuisance parameters (output)
    nuisance_params = OrderedDict()

    # When no systematics setup is defined
    if systematics is None:
        return nuisance_params

    # Parse scale factors from strings in systematics
    logger.debug("Systematics setup: %s", systematics)

    systematics_scales = []
    for key, value in six.iteritems(systematics):
        if key in ["mur", "muf", "mu"]:
            scale_factors = value.split(",")
            scale_factors = [float(sf) for sf in scale_factors]

            if len(scale_factors) == 0:
                raise RuntimeError("Cannot parse scale factor string %s", value)
            elif len(scale_factors) == 1:
                scale_factors = (scale_factors[0],)
            else:
                scale_factors = (scale_factors[-1], scale_factors[0])
            systematics_scales.append(scale_factors)
        else:
            systematics_scales.append(None)

    # Untar and parse LHE file
    initrwgts = _untar_and_parse_lhe_file(filename, ["initrwgt"])

    # Find weight groups
    weight_groups = []
    try:
        for initrwgt in initrwgts:
            weight_groups += initrwgt.findall("weightgroup")
    except KeyError as e:
        raise RuntimeError("Could not find weight groups in LHE file!\n%s", e)

    if len(weight_groups) == 0:
        raise RuntimeError("Zero weight groups in LHE file!")

    # What have we already found?
    systematics_scale_done = []
    for val in systematics_scales:
        if val is None:
            systematics_scale_done.append([True, True])
        elif len(val) == 1:
            systematics_scale_done.append([False, True])
        else:
            systematics_scale_done.append([False, False])

    systematics_pdf_done = False

    # Loop over weight groups and weights and identify benchmarks
    logger.debug("%s weight groups", len(weight_groups))
    for wg in weight_groups:
        logger.debug("Weight group: %s", wg)
        try:
            wg_name = wg.attrib["name"]
        except KeyError:
            logger.warning("Weight group does not have name attribute")
            continue

        if "mg_reweighting" in wg_name.lower():  # Physics reweighting
            logger.debug("Found physics reweighting weight group %s", wg_name)
            continue

        elif (
            "mu" in systematics or "muf" in systematics or "mur" in systematics
        ) and "scale variation" in wg_name.lower():  # Found scale variation weight group
            logger.debug("Found scale variation weight group %s", wg_name)

            weights = wg.findall("weight")

            for weight in weights:
                try:
                    weight_id = str(weight.attrib["id"])
                    weight_muf = float(weight.attrib["MUF"])
                    weight_mur = float(weight.attrib["MUR"])
                except KeyError:
                    logger.warning("Scale variation weight does not have all expected attributes")
                    continue

                logging.debug("Found scale variation weight %s / muf = %s, mur = %s", weight_id, weight_muf, weight_mur)

                # Let's skip the entries with a varied dynamical scale for now
                weight_dynscale = None
                for key in ["dynscale", "dyn_scale", "DYNSCALE", "DYN_SCALE"]:
                    try:
                        weight_dynscale = int(weight.attrib["dynscale"])
                    except KeyError:
                        pass
                if weight_dynscale is not None:
                    continue

                # Matching time!
                for i, (syst_name, syst_scales, syst_done) in enumerate(
                    zip(systematics.keys(), systematics_scales, systematics_scale_done)
                ):
                    if syst_name == "mur":
                        for k in [0, 1]:
                            if (
                                not syst_done[k]
                                and approx_equal(weight_mur, syst_scales[k])
                                and approx_equal(weight_muf, 1.0)
                            ):
                                try:
                                    benchmarks = nuisance_params[syst_name]
                                except KeyError:
                                    benchmarks = [None, None]

                                benchmarks[k] = weight_id
                                nuisance_params[syst_name] = benchmarks

                                systematics_scale_done[i][k] = True
                                break

                    if syst_name == "muf":
                        for k in [0, 1]:
                            if (
                                not syst_done[k]
                                and approx_equal(weight_mur, 1.0)
                                and approx_equal(weight_muf, syst_scales[k])
                            ):
                                try:
                                    benchmarks = nuisance_params[syst_name]
                                except KeyError:
                                    benchmarks = [None, None]

                                benchmarks[k] = weight_id
                                nuisance_params[syst_name] = benchmarks

                                systematics_scale_done[i][k] = True
                                break

                    if syst_name == "mu":
                        for k in [0, 1]:
                            if (
                                not syst_done[k]
                                and approx_equal(weight_mur, syst_scales[k])
                                and approx_equal(weight_muf, syst_scales[k])
                            ):
                                try:
                                    benchmarks = nuisance_params[syst_name]
                                except KeyError:
                                    benchmarks = [None, None]

                                benchmarks[k] = weight_id
                                nuisance_params[syst_name] = benchmarks

                                systematics_scale_done[i][k] = True
                                break

        elif "pdf" in systematics and (
            systematics["pdf"] in wg_name.lower() or "pdf" in wg_name.lower() or "ct" in wg_name.lower()
        ):  # PDF reweighting
            logger.debug("Found PDF variation weight group %s", wg_name)

            weights = wg.findall("weight")

            for i, weight in enumerate(weights):
                try:
                    weight_id = str(weight.attrib["id"])
                    weight_pdf = int(weight.attrib["PDF"])
                except KeyError:
                    logger.warning("Scale variation weight does not have all expected attributes")
                    continue

                logger.debug("Found PDF weight %s / %s", weight_id, weight_pdf)

                # Add every PDF Hessian direction to nuisance parameters
                nuisance_params["pdf_{}".format(i)] = [weight_id, None]

                systematics_pdf_done = True

        else:
            logging.debug("Found other weight group %s", wg_name)

    # Check that everything was found
    if "pdf" in systematics.keys() and not systematics_pdf_done:
        logger.warning(
            "Could not find weights for the PDF uncertainties in LHE file! The most common source of this"
            " error is not having installed LHAPDF with its Python interface. Please make sure that you "
            " have installed this. You can also check the log file produced by MadGraph for a warning"
            " about this. If LHAPDF is correctly installed and you still get this warning, please check"
            " manually whether the LHE file at %s contains weights from PDF variation, and contact"
            " the MadMiner developer team about this. If you continue with the analysis, MadMiner"
            " will disregard PDF uncertainties.",
            filename,
        )

    for syst_name, (done1, done2) in zip(systematics.keys(), systematics_scale_done):
        if not (done1 and done2):
            logger.warning(
                "Did not find benchmarks representing scale variation uncertainty %s in LHE file!", syst_name
            )
            logger.warning(
                "Could not find weights for the scale uncertainty %s in LHE file! The most common source of "
                " this error is not having installed LHAPDF with its Python interface. Please make sure that"
                " you have installed this. You can also check the log file produced by MadGraph for a "
                "warning about this. If LHAPDF is correctly installed and you still get this warning, please"
                " check manually whether the LHE file at %s contains weights from PDF variation, and contact"
                " the MadMiner developer team about this. If you continue with the analysis, MadMiner"
                " will disregard PDF uncertainties.",
                syst_name,
                filename,
            )

    return nuisance_params


def _parse_xml_event(event, sampling_benchmark):
    # Initialize weights and momenta
    weights = OrderedDict()
    particles = []
    global_event_data = {}

    # Split kinematics part in tag line and momenta
    event_text = event.text
    tag_line = None
    particle_lines = []
    for line in event_text.splitlines():
        elements = line.split()
        if len(elements) < 2:
            continue
        if tag_line is None:
            tag_line = elements
        else:
            particle_lines.append(elements)

    # Parse tag
    assert tag_line is not None
    global_event_data["n_particles"] = float(tag_line[0])
    weights[sampling_benchmark] = float(tag_line[2])
    global_event_data["scale"] = float(tag_line[3])
    global_event_data["alpha_qed"] = float(tag_line[4])
    global_event_data["alpha_qcd"] = float(tag_line[5])

    # Parse momenta
    for elements in particle_lines:
        if len(elements) < 10:
            continue
        status = int(elements[1])

        if status == 1:
            pdgid = int(elements[0])
            px = float(elements[6])
            py = float(elements[7])
            pz = float(elements[8])
            e = float(elements[9])
            spin = float(elements[12])
            particle = MadMinerParticle()
            particle.setpxpypze(px, py, pz, e)
            particle.set_pdgid(pdgid)
            particle.set_spin(spin)
            particles.append(particle)

    # Weights
    if event.find("rwgt") is not None:
        for weight in event.find("rwgt").findall("wgt"):
            weight_id, weight_value = weight.attrib["id"], float(weight.text)
            weights[weight_id] = weight_value

    return particles, weights, global_event_data


def _parse_txt_events(filename, sampling_benchmark):
    # Initialize weights and momenta
    weights = OrderedDict()
    particles = []

    # Some tags so that we know where in the event we are
    do_tag = False
    do_momenta = False
    do_reweight = False

    # Event
    n_events = 0

    reset_event = False

    # Loop through lines in Event
    with open(filename, "r") as file:
        for line in file.readlines():
            # Clean up line
            try:
                line = line.split("#")[0]
            except:
                pass
            line = line.strip()
            elements = line.split()

            # Skip empty/commented out lines
            if len(line) == 0 or len(elements) == 0:
                continue

            # End of LHE file
            elif line == "</LesHouchesEvents>":
                return

            # Beginning of event
            elif line == "<event>":
                # Initialize weights and momenta
                weights = OrderedDict()
                particles = []

                # Some tags so that we know where in the event we are
                do_tag = True
                do_momenta = False
                do_reweight = False

            # End of event
            elif line == "</event>":
                n_events += 1
                yield particles, weights

                # Reset weights and momenta
                weights = OrderedDict()
                particles = []

                # Some tags so that we know where in the event we are
                do_tag = False
                do_momenta = False
                do_reweight = False

            # Beginning of unimportant block
            elif line == "<mgrwt>":
                do_tag = False
                do_momenta = False
                do_reweight = False

            # Beginning of weight block
            elif line == "<rwgt>":
                do_tag = False
                do_momenta = False
                do_reweight = True

            # End of weight block
            elif line == "</rwgt>":
                do_tag = False
                do_momenta = False
                do_reweight = False

            # Read tag -> first weight
            elif do_tag:
                weights[sampling_benchmark] = float(elements[2])

                do_tag = False
                do_momenta = True
                do_reweight = False

            # Read Momenta and store as 4-vector
            elif do_momenta:
                status = int(elements[1])
                if status == 1:
                    pdgid = int(elements[0])
                    px = float(elements[6])
                    py = float(elements[7])
                    pz = float(elements[8])
                    e = float(elements[9])
                    spin = float(elements[12])
                    particle = MadMinerParticle()
                    particle.setpxpypze(px, py, pz, e)
                    particle.set_pdgid(pdgid)
                    particle.set_spin(spin)
                    particles.append(particle)

            # Read reweighted weights
            elif do_reweight:
                rwgtid = line[line.find("<") + 1 : line.find(">")].split("=")[1][1:-1]
                rwgtval = float(line[line.find(">") + 1 : line.find("<", line.find("<") + 1)])
                weights[rwgtid] = rwgtval


def _parse_lhe_file_with_bad_chars(filename):
    # In some cases, the LHE comments can contain bad characters
    with open(filename, "r") as file:
        for line in file:
            comment_pos = line.find("#")
            if comment_pos >= 0:
                yield line[:comment_pos]
            else:
                yield line


def _untar_and_parse_lhe_file(filename, tags=None):
    # Untar event file
    new_filename, extension = os.path.splitext(filename)
    if extension == ".gz":
        if not os.path.exists(new_filename):
            call_command("gunzip -c {} > {}".format(filename, new_filename))
        filename = new_filename

    for event, elem in ET.iterparse(filename):
        if tags and elem.tag not in tags:
            continue
        else:
            yield elem

        elem.clear()


def _get_objects(particles, particles_truth, met_resolution=None, global_event_data=None):
    # Find visible particles
    electrons = []
    muons = []
    taus = []
    photons = []
    jets = []
    leptons = []
    neutrinos = []
    unstables = []
    invisibles = []

    for particle in particles:
        pdgid = abs(particle.pdgid)
        if pdgid in [1, 2, 3, 4, 5, 6, 9, 21]:
            jets.append(particle)
        elif pdgid == 11:
            electrons.append(particle)
            leptons.append(particle)
        elif pdgid == 13:
            muons.append(particle)
            leptons.append(particle)
        elif pdgid == 15:
            taus.append(particle)
        elif pdgid == 22:
            photons.append(particle)
        elif pdgid in [12, 14, 16]:
            neutrinos.append(particle)
            invisibles.append(particle)
        elif pdgid in [23, 24, 25]:
            unstables.append(particle)
        else:
            logger.warning("Unknown particle with PDG id %s, treating as invisible!")
            invisibles.append(particle)

    # Sort by pT
    electrons = sorted(electrons, reverse=True, key=lambda x: x.pt)
    muons = sorted(muons, reverse=True, key=lambda x: x.pt)
    taus = sorted(taus, reverse=True, key=lambda x: x.pt)
    photons = sorted(photons, reverse=True, key=lambda x: x.pt)
    leptons = sorted(leptons, reverse=True, key=lambda x: x.pt)
    neutrinos = sorted(neutrinos, reverse=True, key=lambda x: x.pt)
    jets = sorted(jets, reverse=True, key=lambda x: x.pt)

    # Sum over all particles
    ht = 0.0
    visible_sum = MadMinerParticle()
    visible_sum.setpxpypze(0.0, 0.0, 0.0, 0.0)

    for particle in particles:
        pdgid = abs(particle.pdgid)
        if pdgid in [1, 2, 3, 4, 5, 6, 9, 11, 13, 15, 21, 22, 23, 24, 25]:
            visible_sum += particle
            ht += particle.pt

    # Soft noise
    if met_resolution is not None:
        noise_std = met_resolution[0] + met_resolution[1] * ht
        noise_x = np.random.normal(0.0, noise_std, size=None)
        noise_y = np.random.normal(0.0, noise_std, size=None)
    else:
        noise_x = 0.0
        noise_y = 0.0

    # MET
    met_x = -visible_sum.px + noise_x
    met_y = -visible_sum.py + noise_y
    met = MadMinerParticle()
    met.setpxpypze(met_x, met_y, 0.0, (met_x ** 2 + met_y ** 2) ** 0.5)

    # Build objects
    objects = math_commands()
    objects.update(
        {
            "p": particles,
            "p_truth": particles_truth,
            "e": electrons,
            "j": jets,
            "a": photons,
            "mu": muons,
            "tau": taus,
            "l": leptons,
            "met": met,
            "v": neutrinos,
        }
    )

    # Global event_data
    if global_event_data is not None:
        objects.update(global_event_data)

    return objects


def _smear_variable(true_value, resolutions, id):
    """ Adds Gaussian nose to a variable """
    try:
        res = float(resolutions[id][0] + resolutions[id][1] * true_value)

        if res <= 0.0:
            return true_value

        return float(true_value + np.random.normal(0.0, float(res), None))

    except (TypeError, KeyError):
        return true_value


def _smear_particles(particles, energy_resolutions, pt_resolutions, eta_resolutions, phi_resolutions):
    """ Applies smearing function to particles of one event """

    # No smearing if any argument is None
    if energy_resolutions is None or pt_resolutions is None or eta_resolutions is None or phi_resolutions is None:
        return particles

    smeared_particles = []

    for particle in particles:
        pdgid = particle.pdgid

        if (
            pdgid not in six.iterkeys(energy_resolutions)
            or pdgid not in six.iterkeys(pt_resolutions)
            or pdgid not in six.iterkeys(eta_resolutions)
            or pdgid not in six.iterkeys(phi_resolutions)
        ):
            continue

        if None in energy_resolutions[pdgid] and None in pt_resolutions[pdgid]:
            raise RuntimeError("Cannot derive both pT and energy from on-shell conditions!")

        # Minimum energy and pT
        m = particle.m
        min_e = 0.0
        if None in pt_resolutions[pdgid]:
            min_e = m
        min_pt = 0.0

        # Smear four-momenta
        e = None
        if None not in energy_resolutions[pdgid]:
            e = min_e - 1.0
            while e <= min_e:
                e = _smear_variable(particle.e, energy_resolutions, pdgid)
        pt = None
        if None not in pt_resolutions[pdgid]:
            pt = min_pt - 1.0
            while pt <= min_pt:
                pt = _smear_variable(particle.pt, pt_resolutions, pdgid)
        eta = _smear_variable(particle.eta, eta_resolutions, pdgid)
        phi = _smear_variable(particle.phi(), phi_resolutions, pdgid)
        while phi > 2.0 * np.pi:
            phi -= 2.0 * np.pi
        while phi < 0.0:
            phi += 2.0 * np.pi

        # Construct particle
        smeared_particle = MadMinerParticle()

        if None in energy_resolutions[pdgid]:
            # Calculate E from on-shell conditions
            smeared_particle.setptetaphim(pt, eta, phi, particle.m)

        elif None in pt_resolutions[pdgid]:
            # Calculate pT from on-shell conditions
            if e > m:
                pt = (e ** 2 - m ** 2) ** 0.5 / np.cosh(eta)
            else:
                pt = 0.0
            smeared_particle.setptetaphie(pt, eta, phi, e)

        else:
            # Everything smeared manually
            smeared_particle.setptetaphie(pt, eta, phi, e)

        # PDG id (also sets charge)
        smeared_particle.set_pdgid(pdgid)

        smeared_particles.append(smeared_particle)

    return smeared_particles


def get_elementary_pdg_ids():
    ids = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 9, 11, -11, 12, -12, 13, -13, 14, -14, 15, -15, 16, -16]
    ids += [21, 22, 23, 24, -24, 25]
    return ids
