/*
 * Copyright (c) 2016, Peter Anderson <peter.anderson@anu.edu.au>
 *
 * This file is part of Semantic Propositional Image Caption Evaluation
 * (SPICE).
 * 
 * SPICE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * SPICE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.

 * You should have received a copy of the GNU Affero General Public
 * License along with SPICE.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

package edu.anu.spice;

import java.io.IOException;
import java.net.URL;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.cmu.meteor.aligner.SynonymDictionary;
import edu.cmu.meteor.util.Constants;
import edu.stanford.nlp.graph.DirectedMultiGraph;
import edu.stanford.nlp.util.MapFactory;

/**
*
* Based on the SceneGraph in scenegraph-1.0 by Sebastian Schuster.
*
*/

public class SceneGraph implements java.io.Serializable {

	private static final long serialVersionUID = 1L;
	protected final DirectedMultiGraph<SceneGraphNode, SceneGraphRelation> graph;
	protected static final MapFactory<SceneGraphNode, Map<SceneGraphNode, List<SceneGraphRelation>>> outerMapFactory = MapFactory
			.hashMapFactory();
	protected static final MapFactory<SceneGraphNode, List<SceneGraphRelation>> innerMapFactory = MapFactory
			.hashMapFactory();

	protected SynonymDictionary synonyms;
	protected boolean allowConceptMerging;

	public SceneGraph(SynonymDictionary synonyms) {
		this.graph = new DirectedMultiGraph<SceneGraphNode, SceneGraphRelation>(outerMapFactory, innerMapFactory);
		this.synonyms = synonyms;
		this.allowConceptMerging = true;
	}

	public SceneGraph(SynonymDictionary synonyms, boolean mergeSimilarNodes) {
		this(synonyms);
		this.allowConceptMerging = mergeSimilarNodes;
	}

	public void addRelation(String obj1, String obj2, String rel) {
		SceneGraphNode source = getOrAddNodeByString(obj1);
		SceneGraphNode target = getOrAddNodeByString(obj2);
		addRelation(source, target, rel);
	}

	public void addObject(String obj) {
		getOrAddNodeByString(obj);
	}

	public void addAttribute(String obj, String attr) {
		SceneGraphNode source = getOrAddNodeByString(obj);
		SemanticConcept concept = new SemanticConcept(attr, this.getSynSets(attr));
		if (this.allowConceptMerging){
			source.mergeAttribute(concept);
		} else {
			source.addAttribute(concept);
		}
	}

	protected Set<Integer> getSynSets(String concept) {
		concept = concept.trim().toLowerCase().replace(" ", "_");
		Set<Integer> set = new HashSet<Integer>(synonyms.getSynSets(concept));
		set.addAll(synonyms.getStemSynSets(concept));
		return set;
	}

	protected boolean addRelation(SceneGraphNode source, SceneGraphNode target, String relation) {
		String word = relation.replaceAll("^be ", "");
		SemanticConcept concept = new SemanticConcept(word, this.getSynSets(word));
		SceneGraphRelation new_rel = new SceneGraphRelation(source, target, concept);
		for (SceneGraphRelation rel : this.graph.getAllEdges()) {
			if (new_rel.equals(rel)) {
				return false; // don't allow duplicates
			}
		}
		if (this.allowConceptMerging){
			for (SceneGraphRelation rel : this.graph.getAllEdges()) {
				if (new_rel.similarTo(rel)) { // Greedily find first match
					rel.merge(new_rel);
					return true;
				}
			}
		}
		this.graph.add(source, target, new_rel);
		return true;
	}

	protected SceneGraphNode getOrAddNodeByString(String word) {
		SemanticConcept concept = new SemanticConcept(word, this.getSynSets(word));
		return this.getOrAddNode(concept);
	}

	protected SceneGraphNode getOrAddNode(SemanticConcept concept) {
		SceneGraphNode similar_node = null;
		SceneGraphNode new_node = new SceneGraphNode(concept);
		float similarity = 0;
		for (SceneGraphNode node : this.graph.getAllVertices()) {
			if (new_node.getObject().equals(node.getObject())) {
				return node; // ignore attributes in equals
			} else { // merge with the most similar node
				float curr_sim = new_node.similarity(node);
				if (curr_sim > similarity){
					similarity = curr_sim;
					similar_node = node;
				}
			}
		}
		if (this.allowConceptMerging && similar_node != null) {
			similar_node.merge(new_node);
			return similar_node;
		}
		this.graph.addVertex(new_node);
		return new_node;
	}

	public List<SceneGraphRelation> relationList() {
		return this.graph.getAllEdges();
	}

	public Set<SceneGraphNode> nodeList() {
		return this.graph.getAllVertices();
	}

	public String toReadableString() {
		StringBuilder buf = new StringBuilder();
		buf.append(String.format("%-20s%-20s%-20s%n", "source", "reln", "target"));
		buf.append(String.format("%-20s%-20s%-20s%n", "---", "----", "---"));
		for (SceneGraphRelation edge : this.relationList()) {
			buf.append(String.format("%-20s%-20s%-20s%n", edge.getSource(), edge.getRelation(), edge.getTarget()));
		}

		buf.append(String.format("%n%n"));
		buf.append(String.format("%-20s%n", "Nodes"));
		buf.append(String.format("%-20s%n", "---"));

		for (SceneGraphNode node : this.nodeList()) {
			buf.append(String.format("%-20s%n", node));
			for (SemanticConcept attr : node.getAttributes()) {
				buf.append(String.format("  -%-20s%n", attr));
			}
		}
		return buf.toString();
	}

	public static void main(String args[]) {
		SynonymDictionary synonyms;
		URL synDirURL = Constants.DEFAULT_SYN_DIR_URL;
		try {
			URL excFileURL = new URL(synDirURL.toString() + "/english.exceptions");
			URL synFileURL = new URL(synDirURL.toString() + "/english.synsets");
			URL relFileURL = new URL(synDirURL.toString() + "/english.relations");
			synonyms = new SynonymDictionary(excFileURL, synFileURL, relFileURL);
		} catch (IOException ex) {
			throw new RuntimeException("Error: Synonym dictionary could not be loaded (" + synDirURL.toString() + ")");
		}

		boolean[] allowMerge = {true,false};
		for (int i=0; i<allowMerge.length; i++){
			SceneGraph sg = new SceneGraph(synonyms, allowMerge[i]);
			sg.addObject("horse");
			sg.addAttribute("horse", "stone-gray");
			sg.addAttribute("horse", "stone-grey");
			sg.addRelation("horse", "building", "in front of");
			sg.addRelation("building", "car", "behind");
			sg.addAttribute("motorcar", "blue");
			sg.addObject("auto");
			sg.addRelation("automobile", "near", "building");
			sg.addRelation("car", "close", "construction");
			System.out.println(sg.toReadableString());
		}
	}

	public void addTuple(ArrayList<String> tuple) {
		if (tuple.size() == 1){
			this.addObject(tuple.get(0));
		} else if (tuple.size() == 2){
			this.addAttribute(tuple.get(0), tuple.get(1));
		} else if (tuple.size() == 3){
			this.addRelation(tuple.get(0), tuple.get(1), tuple.get(2));
		}
	}

}
