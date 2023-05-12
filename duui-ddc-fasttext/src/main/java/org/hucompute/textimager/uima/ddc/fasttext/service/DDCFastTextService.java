package org.hucompute.textimager.uima.ddc.fasttext.service;

import org.eclipse.microprofile.openapi.annotations.OpenAPIDefinition;
import org.eclipse.microprofile.openapi.annotations.info.Contact;
import org.eclipse.microprofile.openapi.annotations.info.Info;
import org.eclipse.microprofile.openapi.annotations.info.License;

@OpenAPIDefinition(
        info = @Info(
                title = "TextImager DDC fastText Service",
                version = "2.0.0",
                description = "Developed by the Text Technology Lab at the Goethe University in Frankfurt, Germany.",
                termsOfService = "https://textimager.hucompute.org/DDC/#disclaimer",
                contact = @Contact(
                        name = "TTLab Team",
                        url = "https://texttechnologylab.org",
                        email = "text2wiki@texttechnologylab.org"
                ),
                license = @License(
                        name = "AGPL",
                        url = "http://www.gnu.org/licenses/agpl-3.0.en.html"
                )
        )
)
public class DDCFastTextService extends javax.ws.rs.core.Application {
}