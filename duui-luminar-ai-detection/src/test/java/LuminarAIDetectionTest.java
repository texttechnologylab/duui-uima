import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.luminar.AIDetection;

public class LuminarAIDetectionTest {

    public static final String example = "current and upcoming surveys will measure the cosmological parameters with an extremely high accuracy . the primary goal of these observations is to eliminate some of the currently viable cosmological models created to explain the late time accelerated expansion ( either real or only inferred ) . however , most of the statistical tests used in cosmology have a strong requirement : the use of a model to fit the data . these statistical tests are usually based on the normal , or Gaussian , distribution . this distribution is defined by two basic parameters : the mean and the variance ( also referred to as standard deviation or simply σ) . the hypothesis that we test , which is referred to as the Model , is also defined by only two parameters : the F and the ω values ; defined by : F(ω) = σ(ω) 2 - (σF)2. The observed data in the analysis is given by . where s is the square amplitude of the CMB temperature vector and is associated with the angular power spectrum while is the amplitude of the CMB temperature vector and is associated with the angular power spectrum . the observed data in the analysis is given by : where s is the square amplitude of the CMB temperature vector and is associated with the angular power spectrum while is the amplitude of the CMB temperature vector and is associated with the angular power spectrum . it is not very useful when dealing with a theoretical distribution that is not in a normal case, such as the CMB , which is defined by : where is the scalar function describing the temperature distribution , and is defined by . the inverse cumulative density is defined by . so, the inverse probability is simply equal to : The assumption is basically: the data we are observing is normally distributed and our model is exactly correct. we use the data to construct the probability for our hypothesis: when using an iterative method to solve for the initial parameters ( ω , F ) of a statistical model , the method must be based on some form of a";

    @Test
    public void test() throws Exception {

        int iWorkers = 1;
        var ctx = new DUUILuaContext().withJsonLibrary();

        var composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(ctx)
                .withWorkers(iWorkers);

        var dockerDriver = new DUUIDockerDriver();
        var remoteDriver = new DUUIRemoteDriver(1000);
        composer.addDriver(dockerDriver, remoteDriver);

        var useDockerImage = true;
        if (useDockerImage) {
            composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/luminar-seq:latest")
                    .withScale(iWorkers)
                    .build());
        } else {
            composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                    .withScale(iWorkers)
                    .build());
        }

        var jCas = JCasFactory.createText(example, "en");
        composer.run(jCas, "test");
        for (var detection : JCasUtil.select(jCas, AIDetection.class)) {
            System.out.println(detection.getModel() + ":" + detection.getLevel() + ":: " + detection.getBegin() + "-" + detection.getEnd() + "  " + detection.getDetectionScore() + ";");
        }
    }
}