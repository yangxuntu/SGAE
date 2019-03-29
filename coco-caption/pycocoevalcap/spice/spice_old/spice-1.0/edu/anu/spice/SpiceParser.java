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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import edu.cmu.meteor.aligner.SynonymDictionary;
import edu.cmu.meteor.util.Constants;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.scenegraph.SemanticGraphEnhancer;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.semgrex.SemgrexMatcher;
import edu.stanford.nlp.semgraph.semgrex.SemgrexPattern;
import edu.stanford.nlp.trees.UniversalEnglishGrammaticalRelations;
import edu.stanford.nlp.util.CoreMap;

/**
 *
 * SpiceParser is a modified version of the RuleBasedParser in
 * scenegraph-1.0 by Sebastian Schuster. The following changes have been
 * made for use in SPICE:
 * - Plural objects are not duplicated in the scene graph, instead the 
 * numeric modifier is made to be an attribute of the object. 
 * - Objects in a conjunction each receive the sentence relations of the other. 
 * - All nouns are added as objects, even if no relation is found. 
 * - Compound nouns are treated as nouns with an attribute.
 *
 */
public class SpiceParser {

	/* A man is riding a horse. */
	static final SemgrexPattern SUBJ_PRED_OBJ_TRIPLET_PATTERN = SemgrexPattern
			.compile("{}=pred >nsubj {tag:/NNP?S?/}=subj >/(iobj|dobj|nmod:.*)/=objreln {tag:/NNP?S?/}=obj !> cop {}");

	/* A woman is smiling. */
	static final SemgrexPattern SUBJ_PRED_PAIR_PATTERN = SemgrexPattern
			.compile("{}=pred >nsubj {tag:/NNP?S?/}=subj !>/(iobj|dobj|nmod:.*)/ {tag:/NNP?S?/} !>cop {}");

	/* The man is a rider. */
	static final SemgrexPattern COPULAR_PATTERN = SemgrexPattern.compile("{}=pred >nsubj {tag:/NNP?S?/}=subj >cop {}");

	/* A smart woman. */
	static final SemgrexPattern ADJ_MOD_PATTERN = SemgrexPattern.compile("{}=obj >/(amod)/ {}=adj");

	/* The man is tall. */
	static final SemgrexPattern ADJ_PRED_PATTERN = SemgrexPattern.compile("{tag:/J.*/}=adj >nsubj {}=obj");

	/* A woman is in the house. */
	static final SemgrexPattern PP_MOD_PATTERN = SemgrexPattern.compile("{tag:/NNP?S?/}=gov >/nmod:.*/=reln {}=mod");

	/* His watch. */
	static final SemgrexPattern POSS_PATTERN = SemgrexPattern
			.compile("{tag:/NNP?S?/}=gov >/nmod:poss/=reln {tag:/NNP?S?/}=mod");

	/*   */
	static final SemgrexPattern AGENT_PATTERN = SemgrexPattern
			.compile("{tag:/V.*/}=pred >/nmod:agent/=reln {tag:/NNP?S?/}=subj >nsubjpass {tag:/NNP?S?/}=obj ");

	/* A cat sitting in a chair. */
	static final SemgrexPattern ACL_PATTERN = SemgrexPattern
			.compile("{}=subj >acl ({tag:/V.*/}=pred >/(iobj|dobj|nmod:.*)/=objreln {tag:/NNP?S?/}=obj)");

	// Sebastian Schuster - TODO: do something special with nmod:by

	// Several people use laptop computers while sitting around on couches and
	// chairs.
	// A red desk chair has been rolled away from the desk.
	// A red emergency truck has a strong silver guard on the front of its
	// grill.
	// They are preparing the animals for a show.
	// Two engines are visible on the plane.

	// TODO: passives without agent
	// TODO: adverbial modifiers - > potentially (green + light green

	// more spatial relations: in the center of, in the front of,

	/* Any noun in a conjuction. */
	static final SemgrexPattern NOUN_CONJ_PATTERN = SemgrexPattern
			.compile("{tag:/NNP?S?/}=tail >/(conj:and|conj:or)/ {tag:/NNP?S?/}=head");

	/* Compound noun */
	static final SemgrexPattern COMPOUND_NOUN_PATTERN = SemgrexPattern
			.compile("{tag:/NNP?S?/}=tail >/(compound)/ {tag:/NNP?S?/}=head");

	/* Any noun, not compound */
	static final SemgrexPattern NOUN_PATTERN = SemgrexPattern.compile("{tag:/NNP?S?/}=word");


	/* From SemanticGraphEnhancer - Both subject and object or PP are plurals. */
	static final SemgrexPattern PLURAL_SUBJECT_OBJECT_PATTERN = SemgrexPattern.compile("{}=pred >nsubj {tag:/NNP?S/}=subj [ >/(.obj)/ ({tag:/NNP?S/}=obj) |  >/(nmod:((?!agent).)*$)/ ({tag:/NNP?S/}=obj >case {}) ] ");

	/* From SemanticGraphEnhancer - Only subject is plural (either no object or PP exists, or they are singular). */
	static final SemgrexPattern PLURAL_SUBJECT_PATTERN = SemgrexPattern.compile("{tag:/NNP?S/}=subj [ == {$} | <nsubj ({} !>/.obj/ {tag:/NNP?S/} !>/(nmod:((?!agent).)*$)/ ({tag:/NNP?S/} > case {}) )]");

	/* From SemanticGraphEnhancer - Only object is plural (either no subject or it is singular). */
	static final SemgrexPattern PLURAL_OTHER_PATTERN = SemgrexPattern.compile("{tag:/NNP?S/}=word !== {$} !<nsubj {} !</.obj|nmod.*/ ({} >nsubj {tag:/NNP?S/})");


	protected class ProposedTuples {

		public ArrayList<ArrayList<String>> tuples;

		public ProposedTuples(){
			this.tuples = new ArrayList<ArrayList<String>>();
		}

		public void addTuple(IndexedWord subj, IndexedWord obj, String predicate){
			ArrayList<String> tuple = new ArrayList<String>();
			tuple.add(subj.lemma().trim().toLowerCase());
			tuple.add(obj.lemma().trim().toLowerCase());
			tuple.add(predicate);
			this.tuples.add(tuple);
		}

		public void addTuple(IndexedWord subj, IndexedWord pred) {
			ArrayList<String> tuple = new ArrayList<String>();
			tuple.add(subj.lemma().trim().toLowerCase());
			tuple.add(pred.lemma().trim().toLowerCase());
			this.tuples.add(tuple);	
		}

		public void addTuple(IndexedWord word) {
			ArrayList<String> tuple = new ArrayList<String>();
			tuple.add(word.lemma().trim().toLowerCase());
			this.tuples.add(tuple);	
		}

		public void addTuple(IndexedWord head, String string, String predicate) {
			ArrayList<String> tuple = new ArrayList<String>();
			tuple.add(head.lemma().trim().toLowerCase());
			tuple.add(string);
			tuple.add(predicate);
			this.tuples.add(tuple);
		}
	}

	protected final SynonymDictionary synonyms;
	protected Annotator annotator;
	protected LmdbTupleDB db;
	Boolean mergeSimilarNodes;

	public SpiceParser(String dbPath, int numThreads, Boolean mergeSimilarNodes) {
		this.mergeSimilarNodes = mergeSimilarNodes;
		this.annotator = new Annotator(numThreads, 10000);
		if (dbPath != null){
			this.db = new LmdbTupleDB(dbPath);
		} else {
			this.db = null;
		}
		URL synDirURL = Constants.DEFAULT_SYN_DIR_URL;
		try {
			URL excFileURL = new URL(synDirURL.toString() + "/english.exceptions");
			URL synFileURL = new URL(synDirURL.toString() + "/english.synsets");
			URL relFileURL = new URL(synDirURL.toString() + "/english.relations");
			this.synonyms = new SynonymDictionary(excFileURL, synFileURL, relFileURL);
		} catch (IOException ex) {
			throw new RuntimeException("Error: Synonym dictionary could not be loaded (" + synDirURL.toString() + ")");
		}
	}

	protected Map<String, ArrayList<ArrayList<String>>> loadTuplesFromDB(List<String> input, boolean cache){
		// Load any pre-processed captions from the database
		Map<String, ArrayList<ArrayList<String>>> captionTuples = this.db.getTransaction(input);
		ArrayList<String> unparsed = new ArrayList<String>();
		for (String caption: input){
			if (!captionTuples.containsKey(caption)){
				unparsed.add(caption);
			}
		}	
		// Parse and save captions not in database
		if (!unparsed.isEmpty()){
			this.annotator.setInput(unparsed);
			Map<String, ArrayList<ArrayList<String>>> newTuples = new HashMap<String, ArrayList<ArrayList<String>>>();
			Iterator<String> caption = unparsed.iterator();
			while (caption.hasNext()){
				List<Annotation> anns = this.annotator.parseNextBatch();
				assert (!anns.isEmpty());
				Iterator<Annotation> ann = anns.iterator();
				while (caption.hasNext() && ann.hasNext()) {
					ProposedTuples tuples = this.parseAnnotation(ann.next());
					newTuples.put(caption.next(), tuples.tuples);
				}
				if (cache){
					this.db.putTransaction(newTuples);
				}
				captionTuples.putAll(newTuples);
				newTuples.clear();
			}
		}
		return captionTuples;
	}

	protected Map<String, ArrayList<ArrayList<String>>> generateTuples(List<String> input){
		Map<String, ArrayList<ArrayList<String>>> captionTuples = new HashMap<String, ArrayList<ArrayList<String>>>();
		this.annotator.setInput(input);
		Iterator<String> caption = input.iterator();
		while(caption.hasNext()){
			List<Annotation> anns = this.annotator.parseNextBatch();
			assert (!anns.isEmpty());
			Iterator<Annotation> ann = anns.iterator();
			while (caption.hasNext() && ann.hasNext()) {
				ProposedTuples tuples = this.parseAnnotation(ann.next());
				captionTuples.put(caption.next(), tuples.tuples);
			}
		}
		return captionTuples;
	}

	protected Map<String, ArrayList<ArrayList<String>>> loadTuples(List<String> input){
		if (this.db == null){
			return this.generateTuples(input);
		} else {
			return this.loadTuplesFromDB(input, true);
		}
	}

	public List<SceneGraph> parseCaptions(List<String> input, List<Integer> chunks) {
		// Load any pre-processed captions from the database
		Map<String, ArrayList<ArrayList<String>>> captionTuples = this.loadTuples(input);

		// Build scene graphs from tuples (with merging etc)
		List<SceneGraph> sgs = new ArrayList<SceneGraph>();
		Iterator<String> it = input.iterator();
		int captionCount;
		for (Integer chunk: chunks){
			captionCount = 0;
			SceneGraph scene = new SceneGraph(synonyms, this.mergeSimilarNodes);
			while(it.hasNext() && captionCount<chunk){
				ArrayList<ArrayList<String>> tuples = captionTuples.get(it.next());
				assert(tuples != null);
				for (ArrayList<String> tuple: tuples){
					scene.addTuple(tuple);
				}
				captionCount++;
			}
			sgs.add(scene);
		}
		return sgs;
	}

	public List<SceneGraph> parseCaptions(List<String> input) {
		Map<String, ArrayList<ArrayList<String>>> captionTuples = this.loadTuples(input);
		List<SceneGraph> sgs = new ArrayList<SceneGraph>();
		for (String caption: input){
			ArrayList<ArrayList<String>> tuples = captionTuples.get(caption);
			assert(tuples != null);
			SceneGraph scene = new SceneGraph(synonyms, this.mergeSimilarNodes);
			for (ArrayList<String> tuple: tuples){
				scene.addTuple(tuple);
			}
			sgs.add(scene);
		}
		return sgs;
	}

	/**
	 * Attaches particles to the main predicate.
	 */
	protected String getPredicate(SemanticGraph sg, IndexedWord mainPred) {
		if (sg.hasChildWithReln(mainPred, UniversalEnglishGrammaticalRelations.PHRASAL_VERB_PARTICLE)) {
			IndexedWord part = sg.getChildWithReln(mainPred,
					UniversalEnglishGrammaticalRelations.PHRASAL_VERB_PARTICLE);
			return String.format("%s %s", mainPred.lemma().equals("be") ? "" : mainPred.lemma(), part.value());
		}
		return mainPred.lemma();
	}

	/**
	 * Checks if a word has a numerical modifier, and if so adds it as an object
	 * with attribute
	 */
	protected void checkForNumericAttribute(ProposedTuples tuples, SemanticGraph sg, IndexedWord word) {
		if (sg.hasChildWithReln(word, UniversalEnglishGrammaticalRelations.NUMERIC_MODIFIER)) {
			IndexedWord nummod = sg.getChildWithReln(word, UniversalEnglishGrammaticalRelations.NUMERIC_MODIFIER);
			/* Prevent things like "number 5" */
			if (nummod.index() < word.index()) {
				tuples.addTuple(word, nummod);
			}
		} else if (sg.hasChildWithReln(word, SemanticGraphEnhancer.QMOD_RELATION)) {
			IndexedWord qmod = sg.getChildWithReln(word, SemanticGraphEnhancer.QMOD_RELATION);
			tuples.addTuple(word, qmod);
		}
	}

	protected ProposedTuples parseAnnotation(Annotation ann) {
		ProposedTuples tuples = new ProposedTuples();
		ArrayList<SemanticGraph> sgs = new ArrayList<SemanticGraph>();
		for (CoreMap sentence : ann.get(CoreAnnotations.SentencesAnnotation.class)) {
			SemanticGraph sg = sentence
					.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);
			sgs.add(sg);
		}
		for (SemanticGraph sg : sgs) {			
			// Everything from RuleBasedParser except resolvePlurals(sg);
			SemanticGraphEnhancer.processQuanftificationModifiers(sg);
			SemanticGraphEnhancer.collapseCompounds(sg);
			SemanticGraphEnhancer.collapseParticles(sg);
			SemanticGraphEnhancer.resolvePronouns(sg);

			SemgrexMatcher matcher = SUBJ_PRED_OBJ_TRIPLET_PATTERN.matcher(sg);
			while (matcher.find()) {
				IndexedWord subj = matcher.getNode("subj");
				IndexedWord obj = matcher.getNode("obj");
				IndexedWord pred = matcher.getNode("pred");
				String reln = matcher.getRelnString("objreln");
				String predicate = getPredicate(sg, pred);
				if (reln.startsWith("nmod:") && !reln.equals("nmod:poss") && !reln.equals("nmod:agent")) {
					predicate += reln.replace("nmod:", " ").replace("_", " ");
				}
				tuples.addTuple(subj, obj, predicate);
			}

			matcher = ACL_PATTERN.matcher(sg);
			while (matcher.find()) {
				IndexedWord subj = matcher.getNode("subj");
				IndexedWord obj = matcher.getNode("obj");
				IndexedWord pred = matcher.getNode("pred");
				String reln = matcher.getRelnString("objreln");
				String predicate = getPredicate(sg, pred);
				if (reln.startsWith("nmod:") && !reln.equals("nmod:poss") && !reln.equals("nmod:agent")) {
					predicate += reln.replace("nmod:", " ").replace("_", " ");
				}
				tuples.addTuple(subj, obj, predicate);
			}

			SemgrexPattern[] subjPredPatterns = { SUBJ_PRED_PAIR_PATTERN, COPULAR_PATTERN };
			for (SemgrexPattern p : subjPredPatterns) {
				matcher = p.matcher(sg);
				while (matcher.find()) {
					IndexedWord subj = matcher.getNode("subj");
					IndexedWord pred = matcher.getNode("pred");
					if (sg.hasChildWithReln(pred, UniversalEnglishGrammaticalRelations.CASE_MARKER)) {
						IndexedWord caseMarker = sg.getChildWithReln(pred,
								UniversalEnglishGrammaticalRelations.CASE_MARKER);
						String prep = caseMarker.value();
						if (sg.hasChildWithReln(caseMarker,
								UniversalEnglishGrammaticalRelations.MULTI_WORD_EXPRESSION)) {
							for (IndexedWord additionalCaseMarker : sg.getChildrenWithReln(caseMarker,
									UniversalEnglishGrammaticalRelations.MULTI_WORD_EXPRESSION)) {
								prep = prep + " " + additionalCaseMarker.value();
							}
						}
						tuples.addTuple(subj, pred, prep);
					} else {
						if (!pred.lemma().equals("be")) {
							tuples.addTuple(subj, pred);
						}
					}
				}
			}

			matcher = ADJ_MOD_PATTERN.matcher(sg);
			while (matcher.find()) {
				IndexedWord obj = matcher.getNode("obj");
				IndexedWord adj = matcher.getNode("adj");
				tuples.addTuple(obj, adj);
			}

			matcher = ADJ_PRED_PATTERN.matcher(sg);
			while (matcher.find()) {
				IndexedWord obj = matcher.getNode("obj");
				IndexedWord adj = matcher.getNode("adj");
				tuples.addTuple(obj, adj);
			}

			matcher = PP_MOD_PATTERN.matcher(sg);
			while (matcher.find()) {
				IndexedWord gov = matcher.getNode("gov");
				IndexedWord mod = matcher.getNode("mod");
				String reln = matcher.getRelnString("reln");
				String predicate = reln.replace("nmod:", "").replace("_", " ");
				if (predicate.equals("poss") || predicate.equals("agent")) {
					continue;
				}
				tuples.addTuple(gov, mod, predicate);
			}

			matcher = POSS_PATTERN.matcher(sg);
			while (matcher.find()) {
				IndexedWord gov = matcher.getNode("gov");
				IndexedWord mod = matcher.getNode("mod");
				tuples.addTuple(mod, gov, "have");
			}

			matcher = AGENT_PATTERN.matcher(sg);
			while (matcher.find()) {
				IndexedWord subj = matcher.getNode("subj");
				IndexedWord obj = matcher.getNode("obj");
				IndexedWord pred = matcher.getNode("pred");
				tuples.addTuple(subj, obj, getPredicate(sg, pred));
			}

			matcher = PLURAL_SUBJECT_OBJECT_PATTERN.matcher(sg);
			while (matcher.findNextMatchingNode()) {
				IndexedWord subj = matcher.getNode("subj");
				IndexedWord obj = matcher.getNode("obj");
				checkForNumericAttribute(tuples, sg, subj);
				checkForNumericAttribute(tuples, sg, obj);
			}

			matcher = PLURAL_SUBJECT_PATTERN.matcher(sg);
			while (matcher.findNextMatchingNode()) {
				IndexedWord subj = matcher.getNode("subj");
				checkForNumericAttribute(tuples, sg, subj);
			}

			matcher = PLURAL_OTHER_PATTERN.matcher(sg);
			while (matcher.findNextMatchingNode()) {
				IndexedWord word = matcher.getNode("word");
				checkForNumericAttribute(tuples, sg, word);
			}

			matcher = COMPOUND_NOUN_PATTERN.matcher(sg);
			Set<IndexedWord> compoundNouns = new HashSet<IndexedWord>();
			while (matcher.find()) {
				IndexedWord tail = matcher.getNode("tail");
				IndexedWord head = matcher.getNode("head");
				compoundNouns.add(tail);
				compoundNouns.add(head);
				tuples.addTuple(tail, head);
			}

			// Must happen last, since it will reuse existing parts of the scene
			// graph
			matcher = NOUN_CONJ_PATTERN.matcher(sg);
			while (matcher.find()) {
				IndexedWord tail = matcher.getNode("tail");
				IndexedWord head = matcher.getNode("head");
				int original_length = tuples.tuples.size();
				for (int i=0; i<original_length; ++i){
					ArrayList<String> prop = tuples.tuples.get(i);
					if (prop.size() == 3 && prop.get(0).equals(head)){
						tuples.addTuple(head, prop.get(1), prop.get(2));
					}
					if (prop.size() == 3 && prop.get(1).equals(tail)){
						tuples.addTuple(tail, prop.get(1), prop.get(2));
					}
				}
			}

			matcher = NOUN_PATTERN.matcher(sg);
			while (matcher.find()) {
				IndexedWord word = matcher.getNode("word");
				if (!compoundNouns.contains(word)) {
					tuples.addTuple(word);
				}
			}
		}
		return tuples;
	}

	public static void main(String args[]) throws IOException {
		SpiceParser parser = new SpiceParser(null, 1, true);
		if (args.length < 1) {
			System.err.println("Processing from stdin. Enter one sentence per line.");
			System.err.print("> ");
			Scanner scanner = new Scanner(System.in);
			String line;
			while ((line = scanner.nextLine()) != null) {
				List<String> captions = new ArrayList<String>();
				captions.add(line);
				SceneGraph scene = parser.parseCaptions(captions).get(0);
				System.err.println("------------------------");
				System.err.println(scene.toReadableString());
				System.err.println("------------------------");
				System.err.print("> ");
			}
			scanner.close();
		}
	}
}
