package org.texttechnologylab.annotation;

import org.apache.uima.cas.impl.CASImpl;
import org.apache.uima.cas.impl.TypeImpl;
import org.apache.uima.cas.impl.TypeSystemImpl;
import org.apache.uima.cas.impl.TypeSystemUtils;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.JCasRegistry;
import org.apache.uima.jcas.tcas.Annotation;

import java.lang.invoke.CallSite;
import java.lang.invoke.MethodHandle;

public class NeerMatchPrediction extends Annotation {

	@SuppressWarnings("hiding")
	public final static String _TypeName = "org.texttechnologylab.uima.type.neermatch.NeerMatchPrediction";

	@SuppressWarnings("hiding")
	public final static int typeIndexID = JCasRegistry.register(NeerMatchPrediction.class);

	@SuppressWarnings("hiding")
	public final static int type = typeIndexID;

	@Override
	public int getTypeIndexID() {
		return typeIndexID;
	}


	/* *******************
	 *   Feature Offsets *
	 * *******************/

	public final static String _FeatName_document1EntityId = "document1EntityId";
	public final static String _FeatName_document1EntityText = "document1EntityText";
	public final static String _FeatName_document2EntityId = "document2EntityId";
	public final static String _FeatName_document2EntityText = "document2EntityText";
	public final static String _FeatName_score = "score";

	/* Feature Adjusted Offsets */
	private final static CallSite _FC_document1EntityId = TypeSystemImpl.createCallSite(NeerMatchPrediction.class,
		"document1EntityId");
	private final static MethodHandle _FH_document1EntityId = _FC_document1EntityId.dynamicInvoker();
	private final static CallSite _FC_document1EntityText = TypeSystemImpl.createCallSite(NeerMatchPrediction.class,
		"document1EntityText");
	private final static MethodHandle _FH_document1EntityText = _FC_document1EntityText.dynamicInvoker();
	private final static CallSite _FC_document2EntityId = TypeSystemImpl.createCallSite(NeerMatchPrediction.class,
		"document2EntityId");
	private final static MethodHandle _FH_document2EntityId = _FC_document2EntityId.dynamicInvoker();
	private final static CallSite _FC_document2EntityText = TypeSystemImpl.createCallSite(NeerMatchPrediction.class,
		"document2EntityText");
	private final static MethodHandle _FH_document2EntityText = _FC_document2EntityText.dynamicInvoker();
	private final static CallSite _FC_score = TypeSystemImpl.createCallSite(NeerMatchPrediction.class, "score");
	private final static MethodHandle _FH_score = _FC_score.dynamicInvoker();

	protected NeerMatchPrediction() {
		// intentionally empty block
	}

	public NeerMatchPrediction(TypeImpl type, CASImpl casImpl) {
		super(type, casImpl);
		readObject();
	}

	public NeerMatchPrediction(JCas jcas) {
		super(jcas);
		readObject();
	}

	public NeerMatchPrediction(JCas jcas, int begin, int end) {
		super(jcas, begin, end);
		readObject();
	}

	private void readObject() {
		// default - does nothing empty block
	}

	// *--------------*
	// * Feature: document1EntityId

	public String getDocument1EntityId() {
		return _getStringValueNc(wrapGetIntCatchException(_FH_document1EntityId));
	}

	public void setDocument1EntityId(String v) {
		_setStringValueNfc(wrapGetIntCatchException(_FH_document1EntityId), v);
	}

	// *--------------*
	// * Feature: document1EntityText

	public String getDocument1EntityText() {
		return _getStringValueNc(wrapGetIntCatchException(_FH_document1EntityText));
	}

	public void setDocument1EntityText(String v) {
		_setStringValueNfc(wrapGetIntCatchException(_FH_document1EntityText), v);
	}

	// *--------------*
	// * Feature: document2EntityId

	public String getDocument2EntityId() {
		return _getStringValueNc(wrapGetIntCatchException(_FH_document2EntityId));
	}

	public void setDocument2EntityId(String v) {
		_setStringValueNfc(wrapGetIntCatchException(_FH_document2EntityId), v);
	}

	// *--------------*
	// * Feature: document2EntityText

	public String getDocument2EntityText() {
		return _getStringValueNc(wrapGetIntCatchException(_FH_document2EntityText));
	}

	public void setDocument2EntityText(String v) {
		_setStringValueNfc(wrapGetIntCatchException(_FH_document2EntityText), v);
	}

	// *--------------*
	// * Feature: score

	public double getScore() {
		return _getDoubleValueNc(wrapGetIntCatchException(_FH_score));
	}

	public void setScore(double v) {
		_setDoubleValueNfc(wrapGetIntCatchException(_FH_score), v);
	}

}
