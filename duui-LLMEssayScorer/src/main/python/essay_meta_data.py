from typing import Dict
import copy


def get_final_score1(scores1: Dict[str, int], scores2: Dict[str, int]):
    return int(scores1["Overall"]) + int(scores2["Overall"])


prompt_essay_set_1 = """
More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. 

Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.
"""

scoring_rubric_set_1 = {
    "Overall": {
        1: {
            "description": "An undeveloped response that may take a position but offers no more than very minimal support.",
            "typical_elements": [
                "Contains few or vague details.",
                "Is awkward and fragmented.",
                "May be difficult to read and understand.",
                "May show no awareness of audience.",
            ],
            "fine_grained_rubric": "",
        },
        2: {
            "description": "An under-developed response that may or may not take a position.",
            "typical_elements": [
                "Contains only general reasons with unelaborated and/or list-like details.",
                "Shows little or no evidence of organization.",
                "May be awkward and confused or simplistic.",
                "May show little awareness of audience.",

            ],
            "fine_grained_rubric": "",
        },
        3: {
            "description": "A minimally-developed response that may take a position, but with inadequate support and details.",
            "typical_elements": [
                "Has reasons with minimal elaboration and more general than specific details.",
                "Shows some organization.",
                "May be awkward in parts with few transitions.",
                "Shows some awareness of audience.",

            ],
            "fine_grained_rubric": "",
        },
        4: {
            "description": "A somewhat-developed response that takes a position and provides adequate support.",
            "typical_elements": [
                "Has adequately elaborated reasons with a mix of general and specific details.",
                "Shows satisfactory organization.",
                "May be somewhat fluent with some transitional language.",
                "Shows adequate awareness of audience.",

            ],
            "fine_grained_rubric": "",
        },
        5: {
            "description": "A developed response that takes a clear position and provides reasonably persuasive support.",
            "typical_elements": [
                "Has moderately well elaborated reasons with mostly specific details.",
                "Exhibits generally strong organization.",
                "May be moderately fluent with transitional language throughout.",
                "May show a consistent awareness of audience.",
            ],
            "fine_grained_rubric": "",
        },
        6: {
            "description": "A well-developed response that takes a clear and thoughtful position and provides persuasive support.",
            "typical_elements": [
                "Has fully elaborated reasons with specific details.",
                "Exhibits strong organization.",
                "Is fluent and uses sophisticated transitional language.",
                "May show a heightened awareness of audience.",

            ],
            "fine_grained_rubric": "",
        },
    }
}

examples_set_1 = [
    {
        "Overall": 3,
        "explanation": "This is a minimally-developed response with inadequate support and detail. The writer takes the position that computers can be harmful to the eyes and then addresses eye damage to three groups of people (kids, teens, adults). A few specific details are included (sensitive eyes, MySpace), but elaboration is minimal. Some organization is demonstrated but few transitions are used. Overall, the response is sufficiently developed to move into the score point '3' range.",
        "essay": "Dear Editor,\nI think it wouldn't be a good idea for most people. Like let's start with kids, they love games so computer's has a lost of games so they would want one, but it's also a bad side of letting them use the computer to long. See kids have very sensetive eyes just like teenagers, but teens are more grown than the little ones. That's how kid's get their eyes damages. Now teens, were more grown up but we could still get our eyes damaged. It's by using the computer, cause they only go on the computer for myspace and myspace has a lot of typt word's in myspace and that's how teen's get their eye's damaged. Now adults are all grown and ready to do what every they want to do. But they can also damag their eye's. this is why I think using the computer a long time is bad thing.",
    },
    {
        "Overall": 1,
        "explanation": "This is an undeveloped response with few details. The writer states that computers benefit society in many ways and provides a few unelaborated reasons (you learn better; you develop a better memory; show you how to get to a proxy). The response is brief with minimal support; more elaboration is needed for a higher score.o",
        "essay": "Dear, Record Journal\nI think that computers Benefit you in many ways. They help you learn Better. You can Develop a better memmory. they can also show you how to get to a proxy.\nI think all schools should have computers in the classroom.\nSincerely,",
    },
    {
        "Overall": 6,
        "explanation": "This writer states that 'computers are the future, and we need to embrace it' and presents three reasons in support (improvement to hand-eye coordination; instant and endless information; ways to stay in touch with friends). Each reason is fully elaborated with specific details (search engine or database; how to fold a napkin to look like a bunny) and personal examples. Effective word choice (hone, re-assess, at your own leisure) and rhetorical questions (want to find the year of George Washington's death?) contribute to the fluency and specificity of the response. Organization is strong with sophisticated transitional language, and the response is well developed enough for a '6.'",
        "essay": "Dear Readers,\n Recently, I have heard many rumblings about technology and what it is doing our culture. Although some people believe that these changes are for the worse, I can tell you that computers are the future, and we need to embrace it. Computers have many benefits, such as improvement to hand eye, coordination, instant and endless information, and great ways to stay in touch with friends.\n It may surprise you to know that computers are a great way to improve your hand eye coordination. Typing is a valued skill in today's world, and spending time on the computer is the only way to hone it. Schools even have classes these days that are designed solely for typing skills. Don't believe the typing improves and eye coordination? Try writing an essay without looking down at your hands, then reassess your hand eye coordination skills.\n One of my favorite things about the computer is instant information that you're always only a few clicks away from anything you want to know. Want to find the year of George Washington's death? Or perhaps you'd like to know where the longest bridge is located? Any simple question can be answered through the Internet dot the best part is: You don't have to get in your car, drive to the library and then spend hours pouring through books. You can simply lock onto the search engine or database and type in your question. Plus, the Internet can teach you how to do anything Easter is around the corner, and my mom looked up, how to fold the napkin to look like a bunny! As you can see, with the Internet at your fingertips, information is endless.Possibly the best aspect of the computer is how it allows you to keep in touch with your friends who have moved for a way. Although you may never see them in person, you can always use a WebCam should talk them face-to-face through the life chat. Also, you can send them emails which are quicker than letters.unlike phone calls which can be expensive, emails don'tCome up any bills. In addition, you will never have to leave another message again, because you can check your emails at your own leisure.\n You must agree with me: The pros of the computer outweight the con by a ton. Today's modern technology allows you to instantly improve hand, eye, coordination, get information, and connect with friends.technology is advancing, and it is for the better.",
    },
    {
        "Overall": 2,
        "explanation": "This is an under-developed response. The writer states the position that 'computers are just fine' and provides three reasons in support (do things the easier way; do college homework; communicate with family and friends). The first two reasons are unelaborated, but the third includes a brief extension (make up an e-mail address and sign up). There is little evidence of organization, but there is enough development for a '2'.",
        "essay": "Hi my name is @CAPS1. I head about how some people think that computers are affecting peoples health and other things. I personally think that computers are Just fine. computers help people do things the easier way. If someone has to do college homework they can complete it on the computer. I think that family and friends can always communicate. All they have to do is make up an e-mail adress and sign up. And that is why I think that computers are just fine.",
    },
    {
        "Overall": 5,
        "explanation": "This developed response contains three moderately well elaborated reasons that explain why computers do not benefit society (not spending enough time with family; causing harm to nature; not exercising enough). Each reason is elaborated with mostly specific details (family time increases grade level by 15% and reduces crime rates by 38%). The response exhibits generally strong organization with transitional language throughout and is moderately fluent. There is enough specific development for a score of '5'",
        "essay": "Dear @CAPS1,\n Advances and technology don't always benefits a society. People are spending way too much time on their computers. They aren't spending enough time with their family or exercising. Also, while they're under their computers, forest are being cut down. Advances and computers will just make matters worse for society.\n People aren't spending enough time with their families, and computers are the main reason. It is known that people who spend time with family are more successful. In fact, recent studies show that family time increases grade levels by 15%, and reduces crime rates by 38%.However, people who spend too much time alone on the computer are more likely to have worse grades, and/or become a criminal. Computers are Taking away too much family time.\n People who spend a lot of time on the computer aren't realizing the harm that's being done to nature that connecting with nature makes people more caring, but forest are being destroyed. This mass destruction of nature causes many risks. It can affect the air quality, it destroys animal habitats, and it ruins chances for new products and medicines. The destruction of nature must be stopped, and sitting at a computer is just making it worse.\n People aren't exercising enough, but rather look at the computer all day.exercise is key to stay healthy. People who really exercise have a 27% increased chance of getting obese, having hard problems, and getting cancer. On the other hand, people who Exercise 2-4 times a week are proven to live up to 19 years longer than those who don't.People have to stop surfing the web and start running on a treadmill.\n It is proven that advances are simply going to make the lives of people, trees, and animals worse. People are spending time with family, saving nature, or exercising. These things aren't getting done and computers aren't helping at all Internet is used, it should be used for making trips for the family, or finding ways to save nature andn exercise. ",
    },
    {
        "Overall": 4,
        "explanation": "This is a somewhat-developed response in which the writer addresses three positive benefits of computers (better hand-eye coordination; learning about faraway places; chatting with friends and family). Elaboration is an adequate mix of general and specific details (clicking and mouse moving; The Andes of Chile; talk to her with a webcam). Organization is satisfactory, and there is enough development for a '4.'",
        "essay": "Dear @CAPS1,\nSome people say that computers are the entertainment of the future. I agree with this statement that computer teach as many things, and it is great that we get to learn all this information. For one thing, computers, teach people to have better hand eye coordination. I have a friend who has a great coordination from just browsing the web. Dot it makes your hand muscles strongToo from all the clicking and mouse moving that another reason why I think computers are great is that it helps us learn about far away places that if you want to see the Andes of Chile You don't have to actually go there; you can look it up on the web.if you want to know what Venus and Mars surface look like, you can look it up! It saves so much time and energy that way. Lastly, computers, allow you to try online chat with friends and family. I actually have a sister that is studying abroad and ChileRight now, we can talk to here with a WebCam! That's a video camera so we can see and hear it each other. It's awesome! It gives me the feeling that she is close to me that computers have connected the world to one another. It all allows people to see all over the world, and to access information about anything t, I'm a part of that connection. Will you be?",
    },
    {
        "Overall": 1,
        "explanation": "The writer of this undeveloped response believes that computers do not have an effect on people (people can get off their computers anytime they want) but are good to have for a few reasons (you can cash your check, talk to people, get a job). The response is brief with very minimal support; more development is needed for a higher score.",
        "essay": "I think that computers do not have an effect on people cause people can get off they computers anytime they want. But I think computers are good to have. You can cash your check from the computer. You can talk to people. And you also can even get a Job. So I",
    },
    {
        "Overall": 2,
        "explanation": "This writer takes the position that computers are bad for people and provides a few reasons supported with some specific details (damages your eyes, websites and games). However, there is little evidence of organization, and the response is under developed overall. More elaboration is needed for a higher score.",
        "essay": "Do I think computers are bad for people? I think it is because people spent too much time on it then ratter going outside and Exercising point I think computer are addictive because once you go in the computer you don't want to get off point I think its bad because it damages your eyes. This is why its bad for people to use computers a This is why its bad for people to use computers alot.\nI think my spending time on computer they start changing you by not exercising. I think its a intertainment in the computer like the websites that games and other different kind of things.",
    },
    {
        "Overall": 3,
        "explanation": "The writer of this minimally-developed response thinks that computers have a positive effect on people and presents three reasons in support (talking to other people on the internet; learning about far away places; helping with school). Some specific details are provided (MySpace; Egypt or Brazil), but the elaboration is inadequate and often general (help you get information for things). The response shows some organization, but more specific development is needed for a higher score.",
        "essay": "Dear news paper,\n I think that the computers are good. they let you talk to people on line. They let you learn about far away places. Also they can help you with school.\nPeople can talk to other people on the internet, like they can go on myspace or Email their friends or family that live far away. For example i have a friend that lives in Florida and I used to talk to talk to him on mysapce until he moved over here to Connecticut.\nAnother good thing is that you could learn about far away places. like you could go on a fake trip to egypt or Brail. You could see alot of pictures of different places. it would be a nice experience.\nFinally i think computers are good because they can help you with school. They can also do this because they will help you get information for things.\n\nThis is why i think we should use computers.",
    },
    {
        "Overall": 4,
        "explanation": "In this somewhat-developed response, the writer takes the position that children spend too much time on the computer and presents some negative effects while also providing advice to parents on how to help. Elaboration is an adequate mix of general (do something else) and specific details (when it's 85° outside). Organization is satisfactory, though there is minor repetition in the third paragraph, and the response is somewhat fluent. More in-depth and specific development is needed for a higher score.",
        "essay": "Dear advocate, I think the people who spend enough time on the computer that they could be outside playing that, but instead, they sit on a chair, focusing on a computer screen. Yes, sometimes it can be fun on the computer. I go on it, I mean everyone in the world goes on the computer now. But too much of the computer can ruin your life, soWhen it's 85° outside and your On the computer all day it's a waste. when you could be running outside or playing football. But instead, you have to waste a beautiful day to the computer once again, children play games and Their favorite sports player is doing. Instead of sitting at a computer and look how he's doing, they should do the sport. I don't think any sports star would one of their fan sitting at a computer all day looking up how he's doing that they would want them to go outside. I don't think that the parents of the child should force him/her off.They should just encourage them to do something else. If you do encounter him/her maybe children can realize I don't need a computer dot I just thought I had a boring day but I made it that way from sitting on a computer all day. Then they might just go out and play for a little bit. So please advocate can you please accept my article and put it in your newspaper.Because there are too many children on the computer. Maybe this article could change that. So please, can you accept my article.",
    },
    {
        "Overall": 5,
        "explanation": "This developed response contains three moderately well elaborated reasons that explain why computers do not benefit society (not spending enough time with family; causing harm to nature; not exercising enough). Each reason is elaborated with mostly specific details (family time increases grade level by 15% and reduces crime rates by 38%). The response exhibits generally strong organization with transitional language throughout and is moderately fluent. There is enough specific development for a score of '5'.",
        "essay": "Dear newspaper,\n I believe the effects of computers on people are good ones that I think this because, it gives you a chance to look something up fast, get in touch with other users, and also the user could use the computer to educate themselves.\n The effects of computers and people are not always bad. For example, you can use it to look something up fast. For example, my cousin Bob wasn't sure about the local movie time so he went on the Internet and found it in a matter of minutes. But he's not the only one who has found the Internet useful for a quick find.In fact, it was surveyed that 7 out of 10 people use the Internet daily for a quick find that Dr. Peter Griffin says 'Without the Internet, we would be lost'. Do you want people to have a bad effect from not benefiting from using computers? I believe that another good effect from using computers is being able to get in touch with people. Just last week I was in need for help on homework. My friend has a email address so I emailed him my question and then he helped me out. But I'm not the only one who has benefited from being able to get in touch with someone when IWas in need. In fact 90% of computer users today are able to check with friends and get help that Dr. James Tyree says 'Without the use of computers, half of the world would not talk'. Do you want effect from being outside too much?\n Computers benefit our society by being able to educate anyone. Last month there was a special about a man who never went to college. Then there was a online college program that he could do from home. He got his diploma. But he isn't the only one who has been educated on the computer. In fact, 10 out of 20 people have gotten their college diploma over the Internet.School teacher doctor Wellenberger says 'half our world would not be educated if it wasn't for computers'. Do you want our world filled with uneducated people? Some may say Computers can cause obsession, or may cause people to get obese. But people in our society to benefit every day from being able to look something up fast, get in touch with other users, and get educated over the internet. Sincerly, a local society member.",
    },
    {
        "Overall": 6,
        "explanation": "This is a well-developed response in which the writer presents several examples of how 'a computer is a very helpful tool' (communicate with far away friends; ability to type school work; the computer is educational). Each reason is fully elaborated with specific details (less expensive than calling your aunt in Russia; you can backspace and print more than one copy; latest fashion trends and political news). Strong organization is demonstrated through the use of sophisticated transitional language, and the response is developed enough for a '6.'",
        "essay": "Dear fellow readers,\n I have heard about how people think computers are becoming a problem, which they are. But now think about your life without a computer. You won't be able to communicate with your far away friends cheaply, you won't be able to type, and you won't be kept up-to-date.A computer is a very helpful tool and I am here to help you realize it.\n What?! Stacey! Did you say you were moving to Japan? Have you ever walked into a school just to hear that your best friend is moving to some far away land. Maybe not, but it has happened to me and a lot of other people plenty of times that four out of five of my school have had a friend move far away. And they all say that they will never see or hear from them again until they remember emaWith email you can communicate with your next-door neighbor or someone have a across the globe that if you're grounded and your parents took away your phone, don't worry! You still can count on email. Plus, emailing is much less expensive than calling your aunt in Russia. Any sympathetic person would realize that a computer is one helpful tool that you can't live without. There is another way that the computer saves you again too, so all you messy or slow riders just might want to read on.\n Have you ever been in a situation where your teacher announces a big test and tells you to study off your notes? And then you realize your notes are too messy and foregn loking to even read? If you have a computer, then you can just type them up. Two out of three of my friends have messy handwriting so they type on the computer instead. This is so much easier than using a typewriter too. You can backspace and print more than one copy with a computer.Will all of you people who write slow just bring a laptop to class and type your notes that I write slow but I can type faster than I can talk. We've covered that the computers are useful, but they are also educational.\n From the 21st century memories of fun computer games when they were little, but when they didn't know what that their parents bought it because it was educational. The computer has a way that teaches kids by having fun that I know I would rather learn my ABC's by playing a computer game, rather than repeating them over and over in school.The computer is also educational because if I don't understand a question, then I type it in Google search and I get a very well thought out answer. You can be kept up-to-date did the latest fashion, trends and political news too. If I'm watching my favorite movie, and my mom is missing her favorite new show, then she can just look it up on the computer. Well, you can't get more educational than that!\n Now you understand our important the computer is to all people around the world. It helps you communicate with friends, type, and even gives you an education! And now that I am grateful to live, live in century that has computers, and you should be too.",
    },
    {
        "Overall": 2,
        "explanation": "This writer thinks that computers have both negative (bad grades in school) and positive effects (help with homework). Much of the elaboration is either general (they don't understand it) or list-like (taxes, homework, projects for work or school), and the response is under developed overall. Some organization is demonstrated through the use of basic transitions (for example, also), but more elaboration is needed for a higher score.",
        "essay": "Computers have negative and positive consequences. For example kids who play on the computer when they get home instead of doing their homework might have bad grades in school, but the upside to computers is that if they need help because they don't under stand it and their parents don't under stand it, then they can use their computer to help them. Also computer help us with mutipal things like taxes, homework, projects for work or school, and many, many more things. People think that computers are extremely bad but the help you more than anything."
    },
    {
        "Overall": 3,
        "explanation": "This writer takes the position that the effects of computers are different for everyone (it all depends on how you spend your time) and offers suggestions to parents to get their child off the computer (give them a time limit; get security; get them into sports). The response is minimally developed with inadequate support. Stronger organization and more development are needed for a higher score.",
        "essay": "My opinion about the effects of computers is that it all depends on how you spend your time. for exxample I spend my time mostly outside but aslo go on the computer. On the other hand my cuzin is inside 24 hours every day and spends about 8 to 10 hours on the computer a day. There are a verioty of optionns to get your child of the computer. But first the parents have to give them a time limit. if they don't they would be glued to the screen. But on the otherside it is true that you also learn about things like, other People, other countries, technically any thing you would like. But there are certain 'things' you don#t want your kids looking at so get security. That is all i can say but think about what I said and also try to get them into sports.",
    },
    {
        "Overall": 3,
        "explanation": "This writer thinks that computers do not benefit people and offers several examples to support this position. While there are some specific ideas (eating Twinkies in front of the computer, staring mindlessly at a computer), elaboration is mostly general (people are wasting their life) and sometimes repetitive (jogging and biking). The response demonstrates some organization, but the elaboration is minimal overall. More specific development is needed for a higher score.",
        "essay": "Dear W Villager, \nI've been discussing about computer and if they are a benefit or not. Well I say they aren't really a benifit. computer can really help hand eye coordingation, but if you were to go to someone's house there's an 90% chance that if the're on the computer they won't be using it using if for hand eye coordination.\nMost people use computers to Troll or just mess around. What these people could be doing is biking, dogging, lifting weight, playing sports, etc. instead of eating twinkies in front of a computer. There are benifits to a computer people just never really relize them.\nThere are many things someone could be doing besided stareing mindlessly at a computer playing a game like theres no tomorrow. people are wasting there life if all they do is stare at a computer. What did people do before computers how about before electronics, cry about how theres nothing to stare mindlessly at, no they exercised the ran & biked and found some thing that was actually worth while.",
    },
    {
        "Overall": 4,
        "explanation": "This writer takes a position that computers have a positive effect on a society and presents three ways that computers can be helpful. (if you're bored, together, information, to talk to other people). Elaboration is an adequate mix of general (That's a lot of people)And specific details (Go to Google; make your YouTube account and create streams). Even though there is some repetition, the organization is satisfactory with thumb use of transitions. Stronger organization, and more in-depth Development are needed for highest score.",
        "essay": "Many people say that computers don't benefit society. They say computers have no use. I say computers are one of the greatest machines ever invented. You can spend time on it if you're bored. You can use it together information for project that you can even use it to talk to other people. computers have many uses for us.\n Firstly, you can use computers if you're bored. Computers are great if you have nothing to do. You can go to Yahoo and make an account so you get mail. You can go to Google and look at their pictures. You can even go to YouTube, make an account and watch videos.according to a survey, every nine out of 10 kits spend time on the computer. That is a lot of people.\n Secondly, people use it together information.if you have a school project you, you can go to Google, yahoo, or ask.com and gather information.when I have school project that need information on people I use The Internet to find the data. It is sad that eight out of 10 people use the Internet for information. That's over 75%.\n Finally, you can use it to talk to people. If you make friends on Yahoo, you can instant message them I do that on YouTube people make YouTube accounts and create streams.stream is where you can talk to other users and watch videos at the same time. It is fun. People say that nine out of 10 peopleTalk to or instant message people that that is 90% of people. That is a lot of people. I am one of those 90%.\n In conclusion, the computer benefits society. You can use it when you're bored, you together, information, and talk to other people. The computer is a machine I hope stays around for a long time.",
    },
    {
        "Overall": 4,
        "explanation": "The writer of this somewhat-developed response chooses to examine both the positive and negative effects of technology as a whole before concluding that technology is a benefit to society. Elaboration is an adequate mix of general (learn things about places) and specific details (an electric socket overloaded or a computer overheated). Satisfactory organization is demonstrated through the use of transitional language, and the response is somewhat fluent. Many examples are provided, but more in-depth and specific development is needed for a higher score.",
        "essay": "When it comes to technology, there are mainly two different views people have a of it.there are people who feel that people are getting dependent on technology, and think that it's beyond those who think that technology is good currently. They are good things and bad things about technology th I thought it would be best to compare the pros and cons of technologyTo see if it's really more trouble that it's worse, or if it's one of the best things ever.\n First, the good things about technology include communication, knowledge, and entertainment dot before there were cars and phones, it would take a while for people in different places to communicate important use to each other. Now you can get in touch with friends and family by pressing send. Also,The Internet is a great way to learn things about places that, if not for technology, people may have never known existed. And there are lots of people who, like to listen to music, and which movies, and unless they could sing or act they would find it very hard to have Those are just a few of the good things about technology.\n The bad things about technology, to name a few, or how people are getting dependent on technology, or are Not getting enough exercise because they spent all day indoors on the couch watching TV or at the computer sending IMs. Also, many people die because of accidents or housefires occur because an electric socket was overloaded or a computer overheated so much it caught fire. These are just problems with basic technology. Not to mention that people will be forgetting managed by talking on the phone at the dinner table or ignoring someone while watching TV. It's really not the best sing for society.\n In conclusion, the pros and cons of technology, both have some pretty good points, but I believe that technology is good for the world, because people are going into space or research. Try and find curses for diseases. Technology has helped the world way more than hurt it.",
    },
    {
        "Overall": 5,
        "explanation": "In this developed response, the writer takes the position that computers have had a negative effect on society and addresses three ways that computer time is cutting into more important things (exercising; traveling and exploring; family and friends). These reasons are moderately well elaborated with mostly specific details (53% of America's population is overweight) and effective rhetorical questions. The response is fluent and exhibits strong organization. More development is needed for a '6'.",
        "essay": "Have you ever wondered how much time you spend on the computer each day? Have computers ever gotten in the way of the more important things in life? Mostly all people would agree that computers don't always have a benefit to them that even I completely agree that computers Have a very lacking effect on people. This is because the time spent on computers usually cats into time spending exercise, traveling and exploring the world, and especially with their beloved, family and friends.\n First off, many Americans always never get enough time to exercise daily. About 53% of America's population, any age, is overweight. This probably has come to a result, because of the recent advances in technology is such as computers. Would you rather spend your whole life being unfit, while you could be healthy and happy about yourself? I'm not sure about the rest of America, but I would much rather be healthy and fit. Computers and other new technology have made us pick them over, old-fashioned exercising!Healthy is to enjoy life as unhealthy is discouragement in your lifestyle that don't let computers get in the way of your goals and dreams!\n Another reason not to use computers, so often is so you can explore the wonderful planet of earth! There are so many better things you could be doing there are much more exciting than computers! For example, the Grand Canyon. The Grand Canyon is a gorgeous and breathtaking landmark.Instead of researching about this majestic, you can un a computer you can be visiting it. A great deal of people in America have never been I would state. There is so much more to explore these days! Says executive tour guide of the Great Wall of China, Ron Falks. Experts like Ron Know how much time on earth we all have to spend, which is extremely minimal, so he's trying to persuade everyone to travel and explore, before it's too late!\n Last, but definitely not least, if you're on the computer all year long, how will you ever spend extra time with your beloved family and dear friends? Last year, my best friend was supposed to have a family gathering with a relatives from Oregon, who she barely even sees once a year. She refused to go since her parents wouldn't let her on the Internet, the whole day. As simple as that, computers can tear apart families.more than 60% of people around the world to see one side of their family. What is the cause of this? Could it be highly advanced computers and laptops? Of course!\n In conclusion, computers are not beneficial for the people of America, because this time could be spent exercising, traveling the vast world, and spending time with family and friends. Don't let computers get in the way of your life act now!",
    },
]

fine_grained_rubric_essay_set_1 = {
    'details': {
        1: {
            "description": "Contains few or vague details."
        },
        2: {
            "description": "Contains only general reasons with unelaborated and/or list-like details."
        },
        3: {
            "description": "Has reasons with minimal elaboration and more general than specific details."
        },
        4: {
            "description": "Has adequately elaborated reasons with a mix of general and specific details."
        },
        5: {
            "description": "Has moderately well elaborated reasons with mostly specific details."
        },
        6: {
            "description": "Has fully elaborated reasons with specific details."
        },
        'overall_description': "The essay has fully elaborated reasons with specific details."
    },
    'organization': {
        1: {
            "description": "Is awkward and fragmented."
        },
        2: {
            "description": "Shows little or no evidence of organization."
        },
        3: {
            "description": "Shows some organization."
        },
        4: {
            "description": "Shows satisfactory organization."
        },
        5: {
            "description": "Exhibits generally strong organization."
        },
        6: {
            "description": "Exhibits strong organization."
        },
        "overall_description": 'The essay exhibits strong organization.'
    },
    'transition': {
        1: {
            "description": "May be difficult to read and understand."
        },
        2: {
            "description": "May be awkward and confused or simplistic."
        },
        3: {
            "description": "May be awkward in parts with few transitions."
        },
        4: {
            "description": "May be somewhat fluent with some transitional language."
        },
        5: {
            "description": "May be moderately fluent with transitional language throughout."
        },
        6: {
            "description": "Is fluent and uses sophisticated transitional language."
        },
        "overall_description": "The essay is fluent and uses sophisticated transitional language."
    },
    'awareness': {
        1: {
            "description": "May show no awareness of audience."
        },
        2: {
            "description": "May show little awareness of audience."
        },
        3: {
            "description": "Shows some awareness of audience."
        },
        4: {
            "description": "Shows adequate awareness of audience."
        },
        5: {
            "description": "May show a consistent awareness of audience."
        },
        6: {
            "description": "May show a heightened awareness of audience."
        },
        "overall_description": "The essay shows a heightened awareness of audience."
    }
}

fine_grained_rubric_essay_set_1_machine = {
    "Position Clarity": {
        "overall_description": "The response should clearly state a position on the effects of computers on people."
    },
    "Support and Elaboration": {
        "overall_description": "The response should provide reasons and details to support the stated position, with varying levels of specificity and elaboration."
    },
    "Organization": {
        "overall_description": "The response should exhibit a logical structure, with ideas presented in a coherent and orderly manner."
    },
    "Fluency and Transitions": {
        "overall_description": "The response should demonstrate smooth flow and use of transitional language to connect ideas."
    },
    "Audience Awareness": {
        "overall_description": "The response should show an understanding of the audience, tailoring the message to engage and persuade readers effectively."
    },
    "Sophistication": {
        "overall_description": "The response should use sophisticated language and demonstrate a thoughtful and nuanced understanding of the topic."
    },
}

essay_set_1 = {
    "number_of_essays": 1785,
    "average_length": 350.0,
    "final_scores": ["domain1_score"],
    "score_ranges": [(2, 12)],
    "single_evaluator_score_ranges": [(1, 6)],
    "full_score_fn": get_final_score1,
    "prompt": prompt_essay_set_1,
    "scoring_rubric": scoring_rubric_set_1,
    "examples": examples_set_1,
    "fine_grained_rubric": fine_grained_rubric_essay_set_1,
    "fine_grained_rubric_machine": fine_grained_rubric_essay_set_1_machine
}


def get_final_score2(scores1: Dict[str, int], scores2: Dict[str, int]):
    return int(
        scores1["Writing Applications"])  # only this score was considered in taghipour-ng-2016-neural for essay set 2


prompt_essay_set_2 = """
Censorship in the Libraries
"All of us can think of a book that we hope none of our children or any other children have taken off the shelf. But if I have the right to remove that book from the shelf -- that work I abhor -- then you also have exactly the same right and so does everyone else. And then we have no books left on the shelf for any of us." --Katherine Paterson, Author
Write a persuasive essay to a newspaper reflecting your views on censorship in libraries. Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and/or reading.
"""

fine_grained_rubric_wa_score6 = """
## Ideas and Content
Does the writing sample fully accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it
- present a unifying theme or main idea without going off on tangents?
- stay completely focused on topic and task?
Does the writing sample include thorough, relevant, and complete ideas? Does it
- include in-depth information and exceptional supporting details that are fully developed?
- fully explore many facets of the topic?

## Organization
Are the ideas in the writing sample organized logically? Does the writing
- present a meaningful, cohesive whole with a beginning, a middle, and an end (i.e., include an inviting introduction and a strong conclusion)?
- progress in an order that enhances meaning?
- include smooth transitions between ideas, sentences, and paragraphs to enhance meaning of text (i.e., have a clear connection of ideas and use topic sentences)?

## Style
Does the writing sample exhibit exceptional word usage? Does it
- include vocabulary to make explanations detailed and precise, descriptions rich, and actions clear and vivid (e.g., varied word choices, action words, appropriate modifiers, sensory details)?
- demonstrate control of a challenging vocabulary?
Does the writing sample demonstrate exceptional writing technique?
- Is the writing exceptionally fluent?
Does it include varied sentence patterns, including complex sentences?
Does it demonstrate use of writer’s techniques (e.g., literary conventions such as imagery and dialogue and/or literary genres such as humor and suspense)?

## Voice
Does the writing sample demonstrate effective adjustment of language and tone to task and reader? Does it
- exhibit appropriate register (e.g., formal, personal, or dialect) to suit task?
- demonstrate a strong sense of audience?
- exhibit an original perspective (e.g., authoritative, lively, and/or exciting)?
"""
fine_grained_rubric_wa_score5 = """
## Ideas and Content
Does the writing sample fully accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it
- present a unifying theme or main idea without going off on tangents?
- stay focused on topic and task?
Does the writing sample include many relevant ideas? Does it
- provide in-depth information and more than adequate supporting details that are developed?
- explore many facets of the topic?

## Organization
Are the ideas in the writing sample organized logically? Does the writing
- present a meaningful, cohesive whole with a beginning, a middle, and an end (i.e., include a solid introduction and conclusion)?
- progress in an order that enhances meaning of text?
- include smooth transitions (e.g., use topic sentences) between sentences and paragraphs to enhance meaning of text? (Writing may have an occasional lapse.)

## Style
Does the writing sample exhibit very good word usage? Does it
- include vocabulary to make explanations detailed and precise, descriptions rich, and actions clear and vivid?
- demonstrate control of vocabulary?
Does the writing sample demonstrate very good writing technique?
Is the writing very fluent?
Does it include varied sentence patterns, including complex sentences?
Does it demonstrate use of writer’s techniques (e.g., literary conventions such as imagery and dialogue and/or literary genres such as humor and suspense)?

## Voice
Does the writing sample demonstrate effective adjustment of language and tone to task and reader? Does it
- exhibit appropriate register (e.g., formal, personal, or dialect) to suit task?
- demonstrate a sense of audience?
- exhibit an original perspective (e.g., authoritative, lively, and/or exciting)?
"""
fine_grained_rubric_wa_score4 = """
## Ideas and Content
Does the writing sample accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it
- present a unifying theme or main idea? (Writing may include minor tangents.)
- stay mostly focused on topic and task?
Does the writing sample include relevant ideas? Does it
- include sufficient information and supporting details? (Details may not be fully developed; ideas may be listed.)
- explore some facets of the topic?

## Organization
Are the ideas in the writing sample organized logically? Does the writing
- present a meaningful whole with a beginning, a middle, and an end despite an occasional lapse (e.g., a weak introduction or conclusion)?
- generally progress in an order that enhances meaning of text?
- include transitions between sentences and paragraphs to enhance meaning of text? (Transitions may be rough, although some topic sentences are included.)

## Style
Does the writing sample exhibit good word usage? Does it
- include vocabulary that is appropriately chosen, with words that clearly convey the writer’s meaning?
- demonstrate control of basic vocabulary?
Does the writing sample demonstrate good writing technique?
Is the writing fluent?
Does it exhibit some varied sentence patterns, including some complex sentences?
Does it demonstrate an attempt to use writer’s techniques (e.g., literary conventions such as imagery and dialogue and/or literary genres such as humor and suspense)?

## Voice
Does the writing sample demonstrate an attempt to adjust language and tone to task and reader? Does it
- generally exhibit appropriate register (e.g., formal, personal, or dialect) to suit task? (The writing may occasionally slip out of register.)
- demonstrate some sense of audience?
- attempt an original perspective?
"""
fine_grained_rubric_wa_score3 = """
## Ideas and Content
Does the writing sample minimally accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it
- attempt a unifying theme or main idea?
- stay somewhat focused on topic and task?
Does the writing sample include some relevant ideas? Does it
- include some information with only a few details, or list ideas without supporting details?
- explore some facets of the topic?

## Organization
Is there an attempt to logically organize ideas in the writing sample? Does the writing
- have a beginning, a middle, or an end that may be weak or absent?
- demonstrate an attempt to progress in an order that enhances meaning? (Progression of text may sometimes be unclear or out of order.)
- demonstrate an attempt to include transitions? (Are some topic sentences used? Are transitions between sentences and paragraphs weak or absent?)

## Style
Does the writing sample exhibit ordinary word usage? Does it
- contain basic vocabulary, with words that are predictable and common?
- demonstrate some control of vocabulary?
Does the writing sample demonstrate average writing technique?
Is the writing generally fluent?
Does it contain mostly simple sentences (although there may be an attempt at more varied sentence patterns)?
Is it generally ordinary and predictable?

## Voice
Does the writing sample demonstrate an attempt to adjust language and tone to task and reader? Does it
- demonstrate a difficulty in establishing a register (e.g., formal, personal, or dialect)?
- demonstrate little sense of audience?
- generally lack an original perspective?
"""
fine_grained_rubric_wa_score2 = """
## Ideas and Content
Does the writing sample only partially accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it
- attempt a main idea?
- sometimes lose focus or ineffectively display focus?
Does the writing sample include few relevant ideas? Does it
- include little information and few or no details?
- explore only one or two facets of the topic?

## Organization
Is there a minimal attempt to logically organize ideas in the writing sample?
Does the writing have only one or two of the three elements: beginning, middle, and end?
Is the writing sometimes difficult to follow? (Progression of text may be confusing or unclear.)
Are transitions weak or absent (e.g., few or no topic sentences)?

## Style
Does the writing sample exhibit minimal word usage? Does it
- contain limited vocabulary? (Some words may be used incorrectly.)
- demonstrate minimal control of vocabulary?
Does the writing sample demonstrate minimal writing technique?
Does the writing exhibit some fluency?
Does it rely mostly on simple sentences?
Is it often repetitive, predictable, or dull?

## Voice
Does the writing sample demonstrate language and tone that may be inappropriate to task and reader? Does it
- demonstrate use of a register inappropriate to the task (e.g., slang or dialect in a formal setting)?
- demonstrate little or no sense of audience?
- lack an original perspective?
"""
fine_grained_rubric_wa_score1 = """
## Ideas and Content
Does the writing sample fail to accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Is it
- difficult for the reader to discern the main idea?
- too brief or too repetitive to establish or maintain a focus?
Does the writing sample include very few relevant ideas?
Does it include little information with few or no details or unrelated details?
Is it unsuccessful in attempts to explore any facets of the prompt?

## Organization
Are the ideas in the writing sample organized illogically?
Does it have only one or two of the three elements: beginning, middle, or end?
Is it difficult to follow, with the order possibly difficult to discern?
Are transitions weak or absent (e.g., without topic sentences)?

## Style
Does the writing sample exhibit less than minimal word usage? Does it
- contain limited vocabulary, with many words used incorrectly?
- demonstrate minimal or less than minimal control of vocabulary?
Does the writing sample demonstrate less than minimal writing technique? Does it
- lack fluency?
- demonstrate problems with sentence patterns?
- consist of writing that is flat and lifeless?

## Voice
Does the writing sample demonstrate language and tone that may be inappropriate to task and reader? Does it
- demonstrate difficulty in choosing an appropriate register?
- demonstrate a lack of a sense of audience?
- lack an original perspective?
"""

scoring_rubric_set_2 = {
    "Writing Applications": {
        6: {
            "description": "A Score Point 6 paper is rare. It fully accomplishes the task in a thorough and insightful manner and has a distinctive quality that sets it apart as an outstanding performance.",
            "typical_elements": [],
            "fine_grained_rubric": fine_grained_rubric_wa_score6,
        },
        5: {
            "description": "A Score Point 5 paper represents a solid performance. It fully accomplishes the task, but lacks the overall level of sophistication and consistency of a Score Point 6 paper.",
            "typical_elements": [],
            "fine_grained_rubric": fine_grained_rubric_wa_score5,
        },
        4: {
            "description": "A Score Point 4 paper represents a good performance. It accomplishes the task, but generally needs to exhibit more development, better organization, or a more sophisticated writing style to receive a higher score.",
            "typical_elements": [],
            "fine_grained_rubric": fine_grained_rubric_wa_score4,
        },
        3: {
            "description": "A Score Point 3 paper represents a performance that minimally accomplishes the task. Some elements of development, organization, and writing style are weak.",
            "typical_elements": [],
            "fine_grained_rubric": fine_grained_rubric_wa_score3,
        },
        2: {
            "description": "A Score Point 2 paper represents a performance that only partially accomplishes the task. Some responses may exhibit difficulty maintaining a focus. Others may be too brief to provide sufficient development of the topic or evidence of adequate organizational or writing style.",
            "typical_elements": [],
            "fine_grained_rubric": fine_grained_rubric_wa_score2,
        },
        1: {
            "description": "A Score Point 1 paper represents a performance that fails to accomplish the task. It exhibits considerable difficulty in areas of development, organization, and writing style. The writing is generally either very brief or rambling and repetitive, sometimes resulting in a response that may be difficult to read or comprehend.",
            "typical_elements": [],
            "fine_grained_rubric": fine_grained_rubric_wa_score1,
        },
    },
    "Language Conventions": {
        4: {
            "description": "Does the writing sample exhibit a superior command of language skills? \nA Score Point 4 paper exhibits a superior command of written English language conventions. The paper provides evidence that the student has a thorough control of the concepts outlined in the Indiana Academic Standards associated with the student’s grade level. In a Score Point 4 paper, there are no errors that impair the flow of communication. Errors are generally of the first-draft variety or occur when the student attempts sophisticated sentence construction.",
            "typical_elements": [
                "Does the writing sample demonstrate a superior command of capitalization conventions?",
                "Does the writing sample demonstrate a superior command of the mechanics of punctuation?",
                "Does the writing sample demonstrate a superior command of grade-level-appropriate spelling?",
                "Does the writing sample demonstrate a superior command of grammar and Standard English usage?",
                "Does the writing sample demonstrate a superior command of paragraphing?",
                "Does the writing sample demonstrate a superior command of sentence structure by not using run-on sentences or sentence fragments?",

            ],
            "fine_grained_rubric": "",
        },
        3: {
            "description": "Does the writing sample exhibit a good control of language skills? \nIn a Score Point 3 paper, errors are occasional and are often of the first-draft variety; they have a minor impact on the flow of communication.",
            "typical_elements": [
                "Does the writing sample demonstrate a good control of capitalization conventions?",
                "Does the writing sample demonstrate a good control of the mechanics of punctuation?",
                "Does the writing sample demonstrate a good control of grade-level-appropriate spelling?",
                "Does the writing sample demonstrate a good control of grammar and Standard English usage?",
                "Does the writing sample demonstrate a good control of paragraphing?",
                "Does the writing sample demonstrate a good control of sentence structure by only occasionally using run-on sentences or sentence fragments?",
            ],
            "fine_grained_rubric": "",
        },
        2: {
            "description": "Does the writing sample exhibit a fair control of language skills? \nIn a Score Point 2 paper, errors are typically frequent and may occasionally impede the flow of communication.",
            "typical_elements": [
                "Does the writing sample demonstrate a fair control of capitalization conventions?",
                "Does the writing sample demonstrate a fair control of the mechanics of punctuation?",
                "Does the writing sample demonstrate a fair control of grade-level-appropriate spelling?",
                "Does the writing sample demonstrate a fair control of grammar and Standard English usage?",
                "Does the writing sample demonstrate a fair control of paragraphing?",
                "Does the writing sample demonstrate a fair control of sentence structure by frequently using run-on sentences or sentence fragments?",
            ],
            "fine_grained_rubric": "",
        },
        1: {
            "description": "Does the writing sample exhibit a minimal or less than minimal control of language skills? \nIn a Score Point 1 paper, errors are serious and numerous. The reader may need to stop and reread part of the sample and may struggle to discern the writer’s meaning.",
            "typical_elements": [
                "Does the writing sample demonstrate a minimal control of capitalization conventions?",
                "Does the writing sample demonstrate a minimal control of the mechanics of punctuation?",
                "Does the writing sample demonstrate a minimal control of grade-level-appropriate spelling?",
                "Does the writing sample demonstrate a minimal control of grammar and Standard English usage?",
                "Does the writing sample demonstrate a minimal control of paragraphing?",
                "Does the writing sample demonstrate a minimal control of sentence structure by using many run-on sentences or sentence fragments?",

            ],
            "fine_grained_rubric": "",
        },
    },
}

examples_set_2 = [
    {
        "Writing Applications": 3,
        "Language Conventions": 3,
        "essay": "Yes, I think that certain magizines should not be put into libraries due to either bad language or unexceptable photos of grown people.\n A library is for children and teens who come to do research, hang out and study, and to use a computer. Adults on the other hand may come just to get away from the noise but will also read the papers, check out movies, or use the computer. I don’t think it’s right that the children and adults are able to get to check out the same movie because they might get something that is innopropriate because they don’t really understand or a movie that was in the wrong section and is not for them to see due to either incompitent language or sexual activity.\n Personally I think that there should be a seperation between the two. Children or toddlers should have one side and others between the ages of 15-adult should have the other side because I feel that during your teen years you know about the same amount of things as the adults do. I’ve been in quite a few libraries that were not organized. I also feel that if a parent needs to come to the library to use the computer but has her child or children with her; there should be a safe enviorment in there that she can take them to and someone will watch them as she focus’s on what has to be done",
        "explanation": "The following list describes a writing sample (shown on the previous page) that earns a Score Point 3 using the ISTEP+ Writing Applications Rubric.\n This sample\n - minimally accomplishes the task (i.e., write a persuasive essay reflecting your views on censorship in libraries and support it with convincing arguments). - stays mostly focused on the topic.\n - includes a few supporting details (e.g., . . . Children or toddlers should have one side and others between the ages of 15-adult should have the other side because I feel that during your teen years you know about the same amount of things as the adults do.).\n - has a weak introduction, a body that lacks development, and no conclusion (e.g., Yes, I think that certain magizines should not be put into libraries due to either bad language or unexceptable photos of grown people.).\n - attempts some sentence variety (e.g., Adults on the other hand may come just to get away from the noise but will also read the papers, check out movies or use the computer.). - displays appropriate register.\n Note: A Score Point 3 paper represents a performance that minimally accomplishes the task. Some elements of development, organization, and writing style are weak.",
    },
    {
        "Writing Applications": 6,
        "Language Conventions": 4,
        "essay": 'What is freedom of speech? It is the ability to speak out your mind without fear of prosecution – but is that all it is? Is it limited to verbal opinion? Or does every kind of speech count? Do books, music, movies, magazine, newspaper articles, and cartoons come under “speech”? Or are they the targets of the arrow called censorship? Author Katherine Paterson said, “if I have the right to remove that book from the shelf…then you also have exactly the same right and so does everyone else. And then we have no books left on the shelf for any of us.” Opinions can be expressed in several different ways. Vocally expressing one’s views cannot be repressed for once the words are spoken, they cannot be taken back. Books can be retracted; music can be banned; movies can be halted – but how are they any different from verbal expression? Merely because they express points of view through art does not mean that they are not a form of “speech” on the author or musician or film maker’s part. They are his or her way of bringing personal ideas and opinions to the public in an engaging manner, providing entertainment and broadening horizons simultaneously. \nLibraries everywhere have banned countless works under the pretext of their “offensive” nature. J.D. Salinger’s literary masterpiece, The Catcher in the Rye, was the cause of numerous revolts across the nation – parents did not want their children reading it in school, educators were appalled by the gross and perverted aspects of the novel – but today, people describe it as a book that captured the mindset of the youth of America during the 40s perfectly and a story that is “truly American.” What if, 60 years ago, censorship had engulfed the tale of Holden Caulfield forever? What if his story had been burned and doomed to never see the light of the day again? Would America have lost a great novel because of how it “offended” some people? Yes, they would have.\n Censorship is not “bad.” Sometimes, it is necessary. For example, sorting movies according to what age someone can watch them without it being “inappropriate” is beneficial it helps you receive knowledge and the truth about the world when you are truly ready to comprehend it. But banning them because of certain aspects that may not be pleasant to some is not helping anyone. If you do not want your child to read a book you deem “inflammatory,” make it your personal rule to keep them away from it. Libraries can make a different section of those kind of works, require parental consent to gain acces to them, or put a warning on them, but not stifle the creative souls of the minds behind these works.\n Paterson is right. One person may have a problem with a certain book – he raises people against it – the bookis banned. Another woman may not like a song because of negative ideas – she rounds up other critics – the song is banned. A parent may find a certain movie inappropriate and offensive – he creates petitions banning it – the movie is banned. You will never find them in any library anywhere. The idea here is that everyone will have problems with something or the other, but if we start getting rid of those things we ultimately end up with nothing at all. Control, not censor. Do not destroy others opinions based on yours. They may be attacking yours, but by censoring them, you are doing the same thing.',
        "explanation": "The following list describes a writing sample (shown above) that earns a Score Point 6 using the ISTEP+ Writing Applications Rubric. \nThis sample \n- fully accomplishes the task and addresses all specific points of the prompt (i.e., write a persuasive essay reflecting your views on censorship in libraries and support it with convincing arguments).\n- stays completely focused on the topic. \n- provides in-depth information and strong supporting details that are fully developed (e.g., J.D. Salinger’s literary masterpiece, The Catcher in the Rye, was the cause of numerous revolts across the nation – parents did not want their children reading it in school, educators were appalled by the gross and perverted aspects of the novel – but today, people describe it as a book that captured the mindset of the youth of America during the 40s perfectly and a story that is “truly American.”). \n- organizes ideas logically and creates a meaningful, cohesive whole; has an engaging introduction, well-composed middle, and a strong conclusion (e.g., The idea here is that everyone will have problems with something or the other, but if we start getting rid of those things we ultimately end up with nothing at all.). \n- demonstrates very good word usage with excellent writing technique, varying vocabulary throughout the essay (e.g., Libraries everywhere had banned countless works under the pretext of their “offensive” nature.). \n- is fluent and easy to read; the writer includes varied sentence patterns, including complex sentences (e.g., They are his or her way of bringing personal ideas and opinions to the public in an engaging manner, providing entertainment and broadening horizons simultaneously.). \n - displays an appropriate register and effectively adjusts language and tone to the task. \nNote: A Score Point 6 paper is an outstanding performance. It fully accomplishes the task in a thorough and insightful manner and has a distinctive quality that sets it apart.",
    },
    {
        "Writing Applications": 1,
        "Language Conventions": 1,
        "essay": "If some one is offensive over a book or magazine in a librarie then they should stay away from where it is. There are people who would like to learn about it and have to learn about it they will not be able to find it if they start taking books off the shevles just caus it is offensive to some one. that person dont have to read that book or magazine so it should not offend them in any way.",
        "explanation": "The following list describes a writing sample (shown on the previous page) that earns a Score Point 1 using the ISTEP+ Writing Applications Rubric.\n This sample\n - does not accomplish the task (i.e., write a persuasive essay reflecting your views on censorship in libraries and support it with convincing arguments).\n - has little focus.\n - provides very few relevant ideas and less than minimal development (e.g., has no introduction or conclusion).\n - exhibits minimal word usage.\n - demonstrates less than minimal writing technique.\n Note: A Score Point 1 paper represents a performance that fails to accomplish the task. It exhibits considerable difficulty in areas of development, organization, and writing style. The writing is generally either very brief or rambling and repetitive, sometimes resulting in a response that may be difficult to read or comprehend."
    },
    {
        "Writing Applications": 5,
        "Language Conventions": 4,
        "essay": "I believe that certain materials, such as books, music, movies, magazines, etc., should not be removed from the shelves of libraries even if they are found offensive to certain people. Libraries are not only meant for books but all types of informative items there for everyone’s use. If certain people believe that items in libraries are offensive, then they should simply stay away from those items. Reasons that I believe this way are, people need books and magazines as well for research, many people enjoy coming to libraries to listen to music and read magazines if they are not available to them elsewhere, and finally not all people find all the books, music, movies, and magazines offensive.\n To remove offensive items from libraries would make it difficult for people to get information they need. If you were to go to a library for research on a band, and the library didn’t have any magazines that were about the band because certain people found those magazines offensive, you would have a horribly difficult time trying to get deacent information. Magazines, music, and books are in libraries not only for peoples entertainment but for certain information that people want. If you were to remove anything offensive, people would know less about the world around them and libraries would be less popular. So, in the end, it would be a ridiculous idea to remove offensive magazines, books, and music from libraries, because most of those there are for information about certain people or places.\n Many people that I know go to the library to browse through books, listen to music, and read magazines that they do not have access too apart from the library. Now if someone took away the things those people enjoyed, they would have to live without and simply get to watch everyone else outside of the library enjoy that music or book, while they only get to watch from the side. When people are browsing throughout a library and come across something they find offensive, they should simply put down that book or magazine, or turn the music off; no one is making them read or listen to it. With that said, because a few people think that a magazine page is rude, or lyrics to a song are inappropriate, they should not ruin the entertainment for the people who do enjoy listening to that music or reading those magazines.\n Finally, not every single person in the world finds the same things offensive. Certain people love big tattoos, while others think that they are grotesque and rude. Just because a few people don’t like tattoos that doesn’t mean they should be outlawed does it? While in a library, almost everyone could find something that they think should not be on those shelves, but the next person may love that book and think it’s one of the best they’ve ever read. Everyone has a different opinion about everything, and if there are a few people who dislike certain books, magazines, or a certain genre of music, there are most likely a few people who absolutely love that book, or magazine, and love that genre of music. So just because there are people with a negative opinion on certain items, does not mean they should remove them from the shelves of libraries.\n In the end, I believe that they should not remove offensive itmes from the shelves, because they could be there for certain research, or for certain peoples entertainment. Personally I would be upset if they removed the offensive books, magazines, and music from the libraries, because as a teenager that is the genre I am mostly attracted to. I am aware though that the senior citizens of the world, do not find the same things that teenagers like very appropriate, but that’s not a big enough reason to take those items away. So finally, you have to think of what good it would do to take all offensive items off the shelves of libraries, because people like what they like, and simply taking those items off the shelves would not change hardly anything except the popularity of libraries.",
        "explanation": "The following list describes a writing sample (shown on the previous page) that earns a Score Point 5 using the ISTEP+ Writing Applications Rubric. This sample\n - fully accomplishes the task and addresses all specific points of the prompt (i.e., write a persuasive essay reflecting your views on censorship in libraries and support it with convincing arguments).\n - stays focused on the topic. - includes many relevant ideas that are fully developed (e.g., If you were to remove anything offensive, people would know less about the world around them and libraries would be less popular.).\n - is organized logically and cohesively with a clear introduction, developed body, and a conclusion (e.g., To remove offensive items from libraries would make it difficult for people to get information they need.).\n - exhibits more than adequate word usage demonstrating good writing technique (e.g., With that said, because a few people think that a magazine page is rude, or lyrics to a song are inappropriate, they should not ruin the entertainment for the people who do enjoy listening to that music or reading those magazines.).\n - is easy to read; uses varied sentence patterns, including complex sentences (e.g., When people are browsing throughout a library and come across something they find offensive, they should simply put down that book or magazine, or turn the music off; no one is making them read or listen to it.).\n - displays an appropriate register and appropriately adjusts language and tone to the task (e.g., I believe that certain materials, such as books, music, movies, magazines, etc., should not be removed from the shelves of libraries even if they are found offensive to certain people.).\n Note: A Score Point 5 paper represents a solid performance. It fully accomplishes the task, but lacks the overall level of sophistication and consistency of a Score Point 6 paper.",
    },
    {
        "Writing Applications": 2,
        "Language Conventions": 2,
        "essay": "I do think that there are books that should be removed because some people take some things offensive. Some people may think that the book may be wrong because of the religion. And some people may not like what the book, movie, or music is teaching their child.\n People that take the religion to be offensive because the book could say that religion that the person belives is not the right way to go.\n Some parents may take the book offensive because it may contain bad things. These parents may not want their kids reading these types of books but if the library has them the kid could easily check out the book and keep it hidden from thier parents.\n Some parents may not want there kids listenig, watchin, or reading anything that may teach or tell the kid to do things that htye shouldnt be doing.\n I think that there are some books, movies, or music that is inapprpriate for smaller children but may be apprpriate for people that are older it all depends on the book.",
        "explanation": "The following list describes a writing sample (shown on the previous page) that earns a Score Point 2 using the ISTEP+ Writing Applications Rubric.\n This sample\n - minimally accomplishes the task (i.e., write a persuasive essay reflecting your views on censorship in libraries and support it with convincing arguments).\n - exhibits some focus (e.g., I do think that there are books that should be removed because some people take some things offensive.).\n - exhibits minimal organization (e.g., has an introduction, lacks transitions and a developed body, and has no conclusion).\n - provides few supporting details (e.g., Some parents may not want thier kids listenig, watchin, or reading anything that may teach or tell the kid to do things that htye shouldnt be doing.).\n - lacks development of ideas.\n - exhibits minimal word usage and writing techniques (e.g., People that take the religion to be offensive because the book could say that religion that the person belives is not the right way to go.).\n Note: A Score Point 2 paper represents a performance that only partially accomplishes the task. Some responses may exhibit difficulty maintaining a focus; others may be too brief to provide sufficient development of the topic or evidence of adequate organizational or writing style.",
    },
    {
        "Writing Applications": 4,
        "Language Conventions": 4,
        "essay": "Allowing libraries to censor material is wrong. The libraries shouldn’t be allowed to tell you what you can and can’t read. It should be up to the person to decide if the material is against their moral values, so they can decide whether or not they want to read it. The library should allow you to read what you want.\n One way censoring is bad is that you would be forced to read what the library things is right. The library would censor stuff they wouldn’t support and make you read or watch things that they believe in. This would not allow for diverse ideas, and would not fully inform people of both sides of the particular situation. Censoring would cause people to be less unique in the ways they think.\n Another reason is that people should be able to choose what they want to read about with out being influenced by the library. People should have a choice on what they want to read. If they think it’s not good, then they should be able to decide whether they want to read it or not. This makes more of a choice, and allows for more freedom for the people.\n Censoring material would be very constricting on people. They would not be able to read what they want. Allowing censorship would also cause less diverse thinking because everyone would be using the same type of material. Censoring would also make libraries more powerful, because they would only be spreading ideas on what they think is right.",
        "explanation": "The following list describes a writing sample (shown on the previous page) that earns a Score Point 4 using the ISTEP+ Writing Applications Rubric.\n This sample\n - adequately accomplishes the task and addresses all points of the prompt (i.e., write a persuasive essay reflecting your views on censorship in libraries and support it with convincing arguments).\n - stays focused on the topic. - provides some supporting details with some development of those ideas (e.g., This would not allow for diverse ideas and would not fully inform people on both sides of the particular situation.).\n - progresses in a logical order with paragraphs; has a clear introduction, body, and conclusion; uses transitions to show a logical progression of ideas (e.g., One way censoring is bad is that you would be forced to read what the library thinks is right.).\n - exhibits good vocabulary (e.g., Allowing censorship would also cause less diverse thinking because everyone would be using the same type of material.).\n - is easy to read and mostly fluent; the writer uses varied sentence patterns, including some complex sentences (e.g., If they think it’s not good, then they should be able to decide whether they want to read it or not.).\n - displays an appropriate register (e.g., This makes more of a choice and allows for more freedom for the people.).\n Note: A Score Point 4 paper represents a good performance. It accomplishes the task, but generally needs to exhibit more development, better organization, and more sophisticated writing style to receive a higher score.",
    },
]

fine_grained_rubric_essay_set_2 = copy.deepcopy(scoring_rubric_set_2)
fine_grained_rubric_essay_set_2["Writing Applications"][
    "overall_description"] = "The essay fully accomplishes the task in a thorough and insightful manner and has a distinctive quality that sets it apart as an outstanding performance."
fine_grained_rubric_essay_set_2["Language Conventions"][
    "overall_description"] = "The essay the writing sample exhibit a superior command of language skills. The paper provides evidence that the student has a thorough control of the concepts outlined in the Indiana Academic Standards associated with the student's grade level."

fine_grained_rubric_essay_set_2_machine = {
    "Ideas and Content": {
        "overall_description": "Evaluate whether the writing fully accomplishes the task, presents a unifying theme, stays focused, and includes thorough, relevant, and complete ideas with exceptional supporting details"
    },
    "Organization": {
        "overall_description": "Assess the logical organization of ideas, ensuring the writing presents a cohesive whole with a clear beginning, middle, and end, and includes smooth transitions that enhance the meaning of the text"
    },
    "Style": {
        "overall_description": "Examine the word usage, looking for detailed and precise explanations, rich descriptions, varied vocabulary, and control of challenging vocabulary, as well as the use of varied sentence patterns and literary techniques"
    },
    "Voice": {
        "overall_description": "Determine if the writing demonstrates effective adjustment of language and tone to suit the task and audience, exhibits an appropriate register, and shows a strong sense of audience with an original perspective"
    },
    "Language Conventions": {
        "overall_description": "Check for superior command of language skills, including capitalization, punctuation, spelling, grammar, Standard English usage, paragraphing, and sentence structure, ensuring there are no errors that impair the flow of communication"
    }
}

essay_set_2 = {
    "number_of_essays": 1800,
    "average_length": 350.0,
    "final_scores": ["domain1_score", "domain2_score"],
    "score_ranges": [(1, 6), (1, 4)],
    "single_evaluator_score_ranges": [(1, 6), (1, 4)],
    "full_score_fn": get_final_score2,
    "prompt": prompt_essay_set_2,
    "scoring_rubric": scoring_rubric_set_2,
    "examples": examples_set_2,
    "fine_grained_rubric": fine_grained_rubric_essay_set_2,
    "fine_grained_rubric_machine": fine_grained_rubric_essay_set_2_machine
}


def get_final_score3(scores1: Dict[str, int], scores2: Dict[str, int]):
    return int(scores1["Overall"])


prompt_essay_set3 = """
Source: ```
ROUGH ROAD AHEAD: Do Not Exceed Posted Speed Limitby Joe Kurmaskie
FORGET THAT OLD SAYING ABOUT NEVER taking candy from strangers. No, a better piece of advice for the solo cyclist would be, “Never accept travel advice from a collection of old-timers who haven’t left the confines of their porches since Carter was in office.” It’s not that a group of old guys doesn’t know the terrain. With age comes wisdom and all that, but the world is a fluid place. Things change. 
At a reservoir campground outside of Lodi, California, I enjoyed the serenity of an early-summer evening and some lively conversation with these old codgers. What I shouldn’t have done was let them have a peek at my map. Like a foolish youth, the next morning I followed their advice and launched out at first light along a “shortcut” that was to slice away hours from my ride to Yosemite National Park.
They’d sounded so sure of themselves when pointing out landmarks and spouting off towns I would come to along this breezy jaunt. Things began well enough. I rode into the morning with strong legs and a smile on my face. About forty miles into the pedal, I arrived at the first “town.” This place might have been a thriving little spot at one time—say, before the last world war—but on that morning it fit the traditional definition of a ghost town. I chuckled, checked my water supply, and moved on. The sun was beginning to beat down, but I barely noticed it. The cool pines and rushing rivers of Yosemite had my name written all over them. 
Twenty miles up the road, I came to a fork of sorts. One ramshackle shed, several rusty pumps, and a corral that couldn’t hold in the lamest mule greeted me. This sight was troubling. I had been hitting my water bottles pretty regularly, and I was traveling through the high deserts of California in June.
I got down on my hands and knees, working the handle of the rusted water pump with all my strength. A tarlike substance oozed out, followed by brackish water feeling somewhere in the neighborhood of two hundred degrees. I pumped that handle for several minutes, but the water wouldn’t cool down. It didn’t matter. When I tried a drop or two, it had the flavor of battery acid.
The old guys had sworn the next town was only eighteen miles down the road. I could make that! I would conserve my water and go inward for an hour or so—a test of my inner spirit. 
Not two miles into this next section of the ride, I noticed the terrain changing. Flat road was replaced by short, rolling hills. After I had crested the first few of these, a large highway sign jumped out at me. It read: ROUGH ROAD AHEAD: DO NOT EXCEED POSTED SPEED LIMIT.
The speed limit was 55 mph. I was doing a water-depleting 12 mph. Sometimes life can feel so cruel. 
I toiled on. At some point, tumbleweeds crossed my path and a ridiculously large snake—it really did look like a diamondback—blocked the majority of the pavement in front of me. I eased past, trying to keep my balance in my dehydrated state.
The water bottles contained only a few tantalizing sips. Wide rings of dried sweat circled my shirt, and the growing realization that I could drop from heatstroke on a gorgeous day in June simply because I listened to some gentlemen who hadn’t been off their porch in decades, caused me to laugh.
It was a sad, hopeless laugh, mind you, but at least I still had the energy to feel sorry for myself. There was no one in sight, not a building, car, or structure of any kind. I began breaking the ride down into distances I could see on the horizon, telling myself that if I could make it that far, I’d be fi ne.
Over one long, crippling hill, a building came into view. I wiped the sweat from my eyes to make sure it wasn’t a mirage, and tried not to get too excited. With what I believed was my last burst of energy, I maneuvered down the hill.
In an ironic twist that should please all sadists reading this, the building—abandoned years earlier, by the looks of it—had been a Welch’s Grape Juice factory and bottling plant. A sandblasted picture of a young boy pouring a refreshing glass of juice into his mouth could still be seen.
I hung my head.
That smoky blues tune “Summertime” rattled around in the dry honeycombs of my deteriorating brain.
I got back on the bike, but not before I gathered up a few pebbles and stuck them in my mouth. I’d read once that sucking on stones helps take your mind off thirst by allowing what spit you have left to circulate. With any luck I’d hit a bump and lodge one in my throat.
It didn’t really matter. I was going to die and the birds would pick me clean, leaving only some expensive outdoor gear and a diary with the last entry in praise of old men, their wisdom, and their keen sense of direction. I made a mental note to change that paragraph if it looked like I was going to lose consciousness for the last time.
Somehow, I climbed away from the abandoned factory of juices and dreams, slowly gaining elevation while losing hope. Then, as easily as rounding a bend, my troubles, thirst, and fear were all behind me.
GARY AND WILBER’S FISH CAMP—IF YOU WANT BAIT FOR THE BIG ONES, WE’RE YOUR BEST BET!
“And the only bet,” I remember thinking.
As I stumbled into a rather modern bathroom and drank deeply from the sink, I had an overwhelming urge to seek out Gary and Wilber, kiss them, and buy some bait—any bait, even though I didn’t own a rod or reel.
An old guy sitting in a chair under some shade nodded in my direction. Cool water dripped from my head as I slumped against the wall beside him.
“Where you headed in such a hurry?”
“Yosemite,” I whispered.
“Know the best way to get there?”
I watched him from the corner of my eye for a long moment. He was even older than the group I’d listened to in Lodi.
“Yes, sir! I own a very good map.”
And I promised myself right then that I’d always stick to it in the future.
“Rough Road Ahead” by Joe Kurmaskie, from Metal Cowboy, copyright © 1999 Joe Kurmaskie.
```
Write a response that explains how the features of the setting affect the cyclist. In your response, include examples from the essay that support your conclusion.
"""

scoring_rubric_set_3 = {
    "Overall": {
        3: {
            "description": "The response demonstrates an understanding of the complexities of the text.",
            "typical_elements": [
                "Addresses the demands of the question",
                "Uses expressed and implied information from the text",
                "Clarifies and extends understanding beyond the literal",

            ],
            "fine_grained_rubric": "",
        },
        2: {
            "description": "The response demonstrates a partial or literal understanding of the text.",
            "typical_elements": [
                "Addresses the demands of the question, although may not develop all parts equally",
                "Uses some expressed or implied information from the text to demonstrate understanding",
                "May not fully connect the support to a conclusion or assertion made about the text(s)",
            ],
            "fine_grained_rubric": "",
        },
        1: {
            "description": "The response shows evidence of a minimal understanding of the text.",
            "typical_elements": [
                "May show evidence that some meaning has been derived from the text",
                "May indicate a misreading of the text or the question",
                "May lack information or explanation to support an understanding of the text in relation to the question",
            ],
            "fine_grained_rubric": "",
        },
        0: {
            "description": "The response is completely irrelevant or incorrect, or there is no response.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
    },
}

examples_set_3 = [
    {
        "Overall": 1,
        "explanation": "This response demonstrates a minimal understanding of the text. Several ideas are presented (lack of water, hot weather, lack of directions) with some relevant paraphrased text support (When the cyclist was low on water he was almost forced to drink hot water). Although, 'hot weather' is a correct idea, this idea is followed with a misreading (caused him to see a juice factory which was a mirage). A repeated, but unexplained idea is given (The lack of directions made him go farther than he was going). The last idea is partially correct (in the end he found a bait place) and partially incorrect (got directions to get their faster).",
        "essay": "It's a story is a lack of water, hot weather, and lack of directions affected the cyclist. When the cyclist was low on water, he was almost forced to drink hot water. The hot weather cost to see juice factory, which was a marriage. The lack of directions made him go farther than he was going. In the end, he found a bait place and got directions to get their faster.",
    },
    {
        "Overall": 0,
        "explanation": "Although on topic (The cyclist in the stories setting), This response does not show evidence of an understanding of the text or the question.the support presented (Affect the way the character is presented because the setting uses a certain toe that emphasize is the way his action take place) Is vague, and could be in reference to any story. No proof that reading has occurred is demonstrated.",
        "essay": "In my opinion, the cyclist in the story setting effects the way the character is presented, because the setting youth is a certain tone that emphasize this the way his actions take place",
    },
    {
        "Overall": 3,
        "explanation": 'This response demonstrates an understanding of the complexities of the text. The various settings of the story (the morning is cool; he sees a "traditional ghost town"; the sun; the road changes; the water runs out; sun gets hotter) along with how the cyclist is affected (probably worries him; his confidence drops again and again; he fears; morale decreases; loses hope; gains hope; hangs his head in sorrow) is discussed through a balance of text references ("as he rides with strong legs and a smile on his face", "beginning to beat down"; he sees a snake; "dropping from a heatstroke"; the old men he got directions from were wrong; site of the building; building is run down and abandoned) and an implied interpretation of the text (supposed to provide water; probably...because... when...but then...) which moves beyond a literal meaning of the story and question.',
        "essay": "The features of the setting in the story affected the cyclist. At first, the morning is cool as a riots, with strong legs and a smile on his face this setting changes as he sees a traditional ghost town that was supposed to provide water, the sun beginning to be down probably worries him a bitch, but not too bad. When the road changes and he sees assured, rolling Hills, his confident drops again and again when he sees his snake. When the water runs out, he fears dropping From heatstroke, as the son gets harder, his morale decreases because he is also doesn't have much water that he loses hope Everything is that he sees all man he got directions from Wang.when the cyclist gains help it side of the building, the setting effect him that, but then he realizes that the building is rundown and abandoned, and he hangs his head in sorrow. this is how the features of the setting effected the cyclist.",
    },
    {
        "Overall": 2,
        "explanation": "This response demonstrates a partial understanding of the text. Several appropriate ideas concerning how the setting affected the cyclist (becomes exhausted; more and more thirsty; makes him even more hopeless) are discussed. Expressed (The path had snakes, tumbleweeds; he sees an abandoned juice factory) and some implied information (made it hard for a cyclist to make his way through; there is nowhere to get water from; the irony; The setting is perfectly arranged for everything to simply go wrong) shows an understanding that moves beyond a minimal interpretation of the question and text.",
        "essay": "Throw the story, the author becomes exhausted. The path had snakes, tumbleweeds, and almost everything to make it hard for a cyclist to make his way through. He gets more more thirsty, but there's nowhere to get the water from that. Finally, he sees the building and the irony of it being in abandoned juice, factory just makes him even more hopeless, that the setting is perfectly arranged for everything to simply go wrong, but that is what makes the stories so interesting.",
    },
    {
        "Overall": 3,
        "explanation": 'In this response demonstrating an understanding of the complexities of the text, the student extends and clarifies understanding beyond the literal by describing the setting as "as formidable an opponent as the actual workout" (setting becomes harsher and harsher; town, sun beating down, cruel pump, flat terrain gave way to rolling hills, the juice factory). The student explains that "everything is against the cyclist, including nature" and the effects are harsh (sufferings continue, pummeled by his surroundings, hopes were crushed). About the setting and the effects on the cyclist, the student insightfully writes, "All these events are enough to destroy anyones spirit. The cyclist almost gives up hope to accept certain death. He has been ferociously beaten by his very surroundings." The student acknowledges a "twist of fate" as the cyclist encounters his salvation" in a thriving store and experiences "new hope and relief." The student concludes, " ... the cyclist has finally survived his trek through nature.',
        "essay": "The setting seems to be a Fadol and opponent at the actual workout. It seems as if everything is against the cyclist, including nature. As the day progresses, and the cyclists journey continues, the setting becomes harsher and harsher. After passing the first town, the sun was beginning to beat down in need of water, allPump gives him is a terrible substance that is suffering. Continue, increasingly by his surroundings, and his thirst for water. If dehydration was not enough, the flat rain gave way to rolling Hills, which held only punish his legs, more dots, reaching possible salvation, his hopes are crush when he Welsh crave juice factory turns out to be abandoned. All these pans are enough to destroy anyone's spirit that the cyclist almost gives up hope to accept certain death that he has been, beaten by his very surroundings, at appears, as if he is faced to die alone in the blistering here. Although he has read in despair, he still continues on the path to disappointment that in a twist, he encounters his thriving store, where he eats and starts, finally encouraging his salvation, this particular setting brings new hope, and relief to the cyclist, who has finally survive, his track through nature"
    },
    {
        "Overall": 1,
        "explanation": "This response demonstrates a minimal understanding of the text. Two brief but related comments are presented (made it harder for the cycleist to ride; made him very thirsty).",
        "essay": "They're setting mate, it harder for the cyclist to write, made him very thirsty",
    },
    {
        "Overall": 2,
        "explanation": 'This response demonstrates a partial understanding of the text. The student agrees that the setting affects the cyclist in "various ways". These ideas are listed (was full of confidence; the cyclist mood quickly changes from optimistic into desperation; he is releified) and supported through text quotes ("Things began well enough. I rode into the morning with strong legs and a smile on my face.") and paraphrasing (while riding through the pines of Yosemit Park; he pedals through snakes; the cyclist finds water), therefore representing a score level 2 response.',
        "essay": "While readings a story 'Do not exceeded posted speed limit', By Joe Kurmaskie The features of the setting affect, the cyclist and various way start in the beginning of the story. The cyclist was full of confidence while writing through the pints of Joe Park for example paragraph four, lines 1 to 2. 'Thanks again well enough, I rode into the morning with strong legs and a smile on my face.'As a story progresses and the setting changes, the cyclist mood quickly changes from optimistic into desperation. For example, he pedals through snakes, hast, he is relieved when the cyclist finds water. In conclusion, this is how the setting effected the cyclist in the short story.",
    },
    {
        "Overall": 1,
        "explanation": "This response demonstrates a minimal understanding of the text. Discussion surrounding the beginning (at first he was riding in a dessert on a bike with no town in sight) and middle of the story (he finds a town its real old and no signs of humans) is presented through a paraphrase of the text. The next idea is a slight misreading (then he gets dirty hot water which makes him dehydrated). The final sentence is an idea that could be derived from reading (the setting almost caused the cyclist to die), but this idea is not supported.",
        "essay": "The setting of the story affected the cyclist and many bad ways. At first he was riding in a desert on a bike with no town inside. Then he finds a town it's real old and no signs of humans. Then he gets dirty hot water which makes him dehydrated. The setting almost caused the cyclist to die.",
    },
    {
        "Overall": 2,
        "explanation": "This response demonstrates a partial understanding of the text. Many ideas concerning how the setting affected the cyclist are presented (confidence; enthusiasm; determination; his determination shook; he looked forward; but was disappointed; loosing hope; he didn't believe). These ideas are only partially supported through expressed he followed the route given to him by the old men - shortcut; came to a place that was supposed to be a town; 'At some point. majority of the pavement'; saw the fish camp) and implied information (it was desserted and spooky; shows how desserted the place was and how it had been taken over by untamed creatures; he didn't believe it was a fish camp until he got up close) derived from reading 'Rough Road Ahead'.",
        "essay": "The cyclist starts. The journey, was confidence in the old man, enthusiasm, and determination. He followed the route given to him by the old man. The route which was supposed to be a shortcut.\n His determination or the foundations of his determination shook when he came to the place, which was supposed to be a town. It was deserted and spooky. 'At some point, Torness cross my path and ridiculously large snake blocked the majority of the pavement.' Shows how deserted the place was, and how it had been taken over by untamed creatures.\n He looked forward to a town with people, but was disappointed with what he saw, which affected him, almost losing hope, until he saw the fish camp. He even didn't believe that was a fish cam until he got up close."
    },
]

fine_grained_rubric_essay_set_3 = {
    'address the question': {
        3: {
            "description": "Addresses the demands of the question",
        },
        2: {
            "description": "Addresses the demands of the question, although may not develop all parts equally",
        },
        1: {
            "description": "May show evidence that some meaning has been derived from the text"
        },
        0: {
            "description": "The response is completely irrelevant or incorrect, or there is no response.",
        },
        "overall_description": "The essay addresses the demands of the question"
    },
    'uses implied information': {
        3: {
            "description": "Uses expressed and implied information from the text"
        },
        2: {
            "description": "Uses some expressed or implied information from the text to demonstrate understanding",
        },
        1: {
            "description": "May indicate a misreading of the text or the question"
        },
        0: {
            "description": "The response is completely irrelevant or incorrect, or there is no response.",
        },
        "overall_description": 'The essay uses expressed and implied information from the text'
    },
    'clarification beyond literal': {
        3: {
            "description": "Clarifies and extends understanding beyond the literal"
        },
        2: {
            "description": "May not fully connect the support to a conclusion or assertion made about the text(s)",
        },
        1: {
            "description": "May lack information or explanation to support an understanding of the text in relation to the question",
        },
        0: {
            "description": "The response is completely irrelevant or incorrect, or there is no response.",
        },
        "overall_description": "The essay clarifies and extends understanding beyond the literal"
    }
}
fine_grained_rubric_essay_set_3_machine = {
    "Understanding of Text Complexities": {
        "overall_description": "Evaluate whether the response demonstrates a deep comprehension of the text, including its nuances and subtleties"},

    "Addressing the Prompt": {
        "overall_description": "Check if the response fully addresses the demands of the question, covering all parts of the prompt"},

    "Use of Textual Evidence": {
        "overall_description": "Look for the use of both expressed and implied information from the text to support the response"},

    "Extension Beyond Literal": {
        "overall_description": "Assess whether the response goes beyond a literal interpretation to provide deeper insights or connections"},

    "Development of Ideas": {
        "overall_description": "Determine if the response develops its ideas thoroughly and equally, ensuring all parts of the question are addressed"},

    "Connection of Support to Conclusion": {
        "overall_description": "Verify that the textual evidence is effectively connected to the conclusions or assertions made in the response"},

    "Accuracy of Interpretation": {
        "overall_description": "Ensure that the response accurately interprets the text and the question, avoiding any misreadings"},

    "Completeness of Explanation": {
        "overall_description": "Check for a comprehensive explanation that supports the understanding of the text in relation to the question"},
}
essay_set_3 = {
    "number_of_essays": 1726,
    "average_length": 150.0,
    "final_scores": ["domain1_score"],
    "score_ranges": [(0, 3)],
    "single_evaluator_score_ranges": [(0, 3)],
    "full_score_fn": get_final_score3,
    "prompt": prompt_essay_set3,
    "scoring_rubric": scoring_rubric_set_3,
    "examples": examples_set_3,
    "fine_grained_rubric": fine_grained_rubric_essay_set_3,
    "fine_grained_rubric_machine": fine_grained_rubric_essay_set_3_machine
}


def get_final_score4(scores1: Dict[str, int], scores2: Dict[str, int]):
    return int(scores1["Overall"])


prompt_essay_set4 = """
Source: ```
Winter Hibiscus by Minfong Ho
Saeng, a teenage girl, and her family have moved to the United States from Vietnam. As Saeng walks home after failing her driver’s test, she sees a familiar plant. Later, she goes to a florist shop to see if the plant can be purchased.
It was like walking into another world. A hot, moist world exploding with greenery. Huge flat leaves, delicate wisps of tendrils, ferns and fronds and vines of all shades and shapes grew in seemingly random profusion.
“Over there, in the corner, the hibiscus. Is that what you mean?” The florist pointed at a leafy potted plant by the corner. 
There, in a shaft of the wan afternoon sunlight, was a single blood-red blossom, its five petals splayed back to reveal a long stamen tipped with yellow pollen. Saeng felt a shock of recognition so intense, it was almost visceral.1
“Saebba,” Saeng whispered.
A saebba hedge, tall and lush, had surrounded their garden, its lush green leaves dotted with vermilion flowers. And sometimes after a monsoon rain, a blossom or two would have blown into the well, so that when she drew the well water, she would find a red blossom floating in the bucket.
Slowly, Saeng walked down the narrow aisle toward the hibiscus. Orchids, lanna bushes, oleanders, elephant ear begonias, and bougainvillea vines surrounded her. Plants that she had not even realized she had known but had forgotten drew her back into her childhood world.
When she got to the hibiscus, she reached out and touched a petal gently. It felt smooth and cool, with a hint of velvet toward the center—just as she had known it would feel.
And beside it was yet another old friend, a small shrub with waxy leaves and dainty flowers with purplish petals and white centers. “Madagascar periwinkle,” its tag announced. How strange to see it in a pot, Saeng thought. Back home it just grew wild, jutting out from the cracks in brick walls or between tiled roofs.
And that rich, sweet scent—that was familiar, too. Saeng scanned the greenery around her and found a tall, gangly plant with exquisite little white blossoms on it.  “Dok Malik,” she said, savoring the feel of the word on her tongue, even as she silently noted the English name on its tag, “jasmine.”
One of the blossoms had fallen off, and carefully Saeng picked it up and smelled it. She closed her eyes and breathed in, deeply. The familiar fragrance filled her lungs, and Saeng could almost feel the light strands of her grandmother’s long gray hair, freshly washed, as she combed it out with the fine-toothed buffalo-horn comb. And when the sun had dried it, Saeng would help the gnarled old fingers knot the hair into a bun, then slip a dok Malik bud into it.
Saeng looked at the white bud in her hand now, small and fragile. Gently, she closed her palm around it and held it tight. That, at least, she could hold on to. But where was the fine-toothed comb? The hibiscus hedge? The well? Her gentle grandmother? 
A wave of loss so deep and strong that it stung Saeng’s eyes now swept over her. A blink, a channel switch, a boat ride into the night, and it was all gone. Irretrievably, irrevocably gone.
And in the warm moist shelter of the greenhouse, Saeng broke down and wept.
It was already dusk when Saeng reached home. The wind was blowing harder, tearing off the last remnants of green in the chicory weeds that were growing out of the cracks in the sidewalk. As if oblivious to the cold, her mother was still out in the vegetable garden, digging up the last of the onions with a rusty trowel. She did not see Saeng until the girl had quietly knelt down next to her.
Her smile of welcome warmed Saeng. “Ghup ma laio le? You’re back?” she said cheerfully. “Goodness, it’s past five. What took you so long? How did it go? Did you—?” Then she noticed the potted plant that Saeng was holding, its leaves quivering in the wind.
Mrs. Panouvong uttered a small cry of surprise and delight. “Dok faeng-noi!” she said. “Where did you get it?”
“I bought it,” Saeng answered, dreading her mother’s next question.
“How much?”
For answer Saeng handed her mother some coins.
“That’s all?” Mrs. Panouvong said, appalled, “Oh, but I forgot! You and the
Lambert boy ate Bee-Maags . . . .”
“No, we didn’t, Mother,” Saeng said.
“Then what else—?”
“Nothing else. I paid over nineteen dollars for it.”
“You what?” Her mother stared at her incredulously. “But how could you? All the seeds for this vegetable garden didn’t cost that much! You know how much we—” She paused, as she noticed the tearstains on her daughter’s cheeks and her puffy eyes.
“What happened?” she asked, more gently.
“I—I failed the test,” Saeng said.
For a long moment Mrs. Panouvong said nothing. Saeng did not dare look her mother in the eye. Instead, she stared at the hibiscus plant and nervously tore off a leaf, shredding it to bits.
Her mother reached out and brushed the fragments of green off Saeng’s hands. “It’s a beautiful plant, this dok faeng-noi,” she finally said. “I’m glad you got it.”
“It’s—it’s not a real one,” Saeng mumbled.
“I mean, not like the kind we had at—at—” She found that she was still too shaky to say the words at home, lest she burst into tears again. “Not like the kind we had before,” she said.
“I know,” her mother said quietly. “I’ve seen this kind blooming along the lake. Its flowers aren’t as pretty, but it’s strong enough to make it through the cold months here, this winter hibiscus. That’s what matters.”
She tipped the pot and deftly eased the ball of soil out, balancing the rest of the plant in her other hand. “Look how root-bound it is, poor thing,” she said. “Let’s plant it, right now.”
She went over to the corner of the vegetable patch and started to dig a hole in the ground. The soil was cold and hard, and she had trouble thrusting the shovel into it. Wisps of her gray hair trailed out in the breeze, and her slight frown deepened the wrinkles around her eyes. There was a frail, wiry beauty to her that touched Saeng deeply.
“Here, let me help, Mother,” she offered, getting up and taking the shovel away from her.
Mrs. Panouvong made no resistance. “I’ll bring in the hot peppers and bitter melons, then, and start dinner. How would you like an omelet with slices of the bitter melon?”
“I’d love it,” Saeng said.
Left alone in the garden, Saeng dug out a hole and carefully lowered the “winter hibiscus” into it. She could hear the sounds of cooking from the kitchen now, the beating of eggs against a bowl, the sizzle of hot oil in the pan. The pungent smell of bitter melon wafted out, and Saeng’s mouth watered. It was a cultivated taste, she had discovered—none of her classmates or friends, not even Mrs. Lambert, liked it—this sharp, bitter melon that left a golden aftertaste on the tongue. But she had grown up eating it and, she admitted to herself, much preferred it to a Big Mac.
The “winter hibiscus” was in the ground now, and Saeng tamped down the soil around it. Overhead, a flock of Canada geese flew by, their faint honks clear and—yes—familiar to Saeng now. Almost reluctantly, she realized that many of the things that she had thought of as strange before had become, through the quiet repetition of season upon season, almost familiar to her now. Like the geese. She lifted her head and watched as their distinctive V was etched against the evening sky, slowly fading into the distance.
When they come back, Saeng vowed silently to herself, in the spring, when the snows melt and the geese return and this hibiscus is budding, then I will take that test again.
“Winter Hibiscus” by Minfong Ho, copyright © 1993 by Minfong Ho, from Join In, Multiethnic Short Stories, by Donald R. Gallo, ed.
```
Read the last paragraph of the story.

"When they come back, Saeng vowed silently to herself, in the spring, when the snows melt and the geese return and this hibiscus is budding, then I will take that test again." 

Write a response that explains why the author concludes the story with this paragraph. In your response, include details and examples from the story that support your ideas.
"""

scoring_rubric_set_4 = {
    "Overall": {
        3: {
            "description": "The response demonstrates an understanding of the complexities of the text.",
            "typical_elements": [
                "Addresses the demands of the question",
                "Uses expressed and implied information from the text",
                "Clarifies and extends understanding beyond the literal",
            ],
            "fine_grained_rubric": "",
        },
        2: {
            "description": "The response demonstrates a partial or literal understanding of the text.",
            "typical_elements": [
                "Addresses the demands of the question, although may not develop all parts equally",
                "Uses some expressed or implied information from the text to demonstrate understanding",
                "May not fully connect the support to a conclusion or assertion made about the text(s)",
            ],
            "fine_grained_rubric": "",
        },
        1: {
            "description": "The response shows evidence of a minimal understanding of the text.",
            "typical_elements": [
                "May show evidence that some meaning has been derived from the text",
                "May indicate a misreading of the text or the question",
                "May lack information or explanation to support an understanding of the text in relation to the question",
            ],
            "fine_grained_rubric": "",
        },
        0: {
            "description": "The response is completely irrelevant or incorrect, or there is no response.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
    },
}

fine_grained_rubric_essay_set_4 = copy.deepcopy(fine_grained_rubric_essay_set_3)  # they are exactly the same!
fine_grained_rubric_essay_set_4_machine = copy.deepcopy(fine_grained_rubric_essay_set_3_machine)

examples_set_4 = [
    {
        "Overall": 1,
        "explanation": "This response shows evidence of a minimal understanding of the text. The writer states that the story's ending is appropriate and presents several ideas concerning events that occur or will occur in the future. A stronger connection, as to why the author concludes the story with the specific paragraph, would be needed for a higher score.",
        "essay": "The author ended the story exist because it is most appropriate. Saenq's Mother gave her the courage to try again. Her mother was not mad, but believed in her. This gave Saenq The strength to try again. Saenq Is showing that she is not going to give up because she failed one that she is going to go back and succed.",
    },
    {
        "Overall": 0,
        "explanation": "This response shows no evidence of understanding of the text. Although on topic, the information given is incorrect (maybe she was leaving and coming back to take the test), a rewording of the prompt (when the buds are bluming), ard/or irrelevant (she is going on a vacation).",
        "essay": "He wrote that ending because maybe she was leaving and coming back to take the test Dutch or she is going on a vacation and coming back when the buds are bluming.",
    },
    {
        "Overall": 3,
        "explanation": "Although concise, this response demonstrates an understanding of the complexities of the text. The theme (to show that she has come full circle) is discussed through use of paraphrased text ideas (That will be the same time of year; the flower will remind her of home; the geese and hibiscus is more familiar to her; it will be spring). These ideas are extended by use of reasoned conclusions (giving her confidence to succeed and pass the test; helping her to adjust to life; when she feels at home and more comfortable) and several relevant comparisons (spring symbolizes rebirth; a new person - new society - new sense of belonging) which clarifies understanding beyond the literal interpretation of the text and questions.",
        "essay": "The author concludes with this statement to show that she has come full circle. That will be the same time of the year that it is now. Also, The flower will remind her of Home🏡, giving her the confidence to succeed and path the test. Now, she says that she and the hibiscus is more familiar to her, and helping her to adjust to her life. When she feels at home and more comfortable, she will be able to pass the test. Although, it seemed that that time of the year will be spring.Spring symbolizes reapers so it fits that she will be trying again as the new person who is used to her new society and feels a new sense of belonging so the girl going to take the test will be a new person."
    },
    {
        "Overall": 2,
        "explanation": "This response demonstrates a partial understanding of the text. Several appropriate reasons, as to why the author concludes the story with the specific paragraph, are presented through text references (vows to take the test again; adapt in her new surroundings) and implied information (shows how Saeng is slowly overcoming her weakness; she is looking forward to the future which is contrary to her nostalgic attitude earlier in the story; shows hope and leaves the reader feeling confident).",
        "essay": "The author concludes the story with this paragraph, because it shows how Saeng Is slowly overcoming her weakness in this new surrounding. She wows to take the test again and is looking forward to the future, which is contrary to her nostalgic attitude earlier in the story. This last paragraph shows hope for Saeng and leave the raider feeling confident in her ability to adapt in her new surroundings.",
    },
    {
        "Overall": 1,
        "explanation": "This response shows evidence of a minimal understanding of the text. The writer recopies the last paragraph and then gives a relevant, but general idea (to show the reader that if you fail, give it some time and then try to succeed, don't just give up forever).",
        "essay": "In the story 'Winter Hibiscus' by Minfong Ho, he ends it with 'when they come back, saeng vowed silently to herself, in the spring, when the snows melt and the geese return and this hibiscus is budding, then I will take that test again.' My opinion on why I think the author ended it like that is to show the reader that if you fail, give it some time and then try to succeed, don't just give up forever.",
    },
    {
        "Overall": 3,
        "explanation": "This response demonstrates an understanding of the complexities of the text. The main reason (it shows Saeng's will to start over but not so quickly) is supported with expressed information (she admired nature in her home and in a pot; the geese will return and her flower will bloom; pass the driving test; the snow melting). These ideas are extended through evaluation of the situation (she seems very spiritual; she sees spring as a starting over time; she feels this would be a good time for her to start over as well) and appropriate comparison (she sees herself as a flower blossoming; she too will be revived and complete her test in spring - the season of revival).",
        "essay": "Spring is the season of reverse that I think the author shows this time to end the story because it shows Saengs will to start over, but not so quickly.\n Saeng seems very spiritual because of the way she admired nature both in her homeland and in a pot. Conclude that she also sees spring as a starting ove time. The geese will return and her flower will bloom and she feels thing would be a good time for her to start over as well.\n I thinkg Saeng too sees herself as a flower - blossoming into a new country and so spring is her time to bloom and pass the driving test. This is why I think the article concluded the story with a line that likes the cheese, returning the flower, blossoming the the snow melting - Saegn too will revive and complete her test in the spring - the season of revival."
    },
    {
        "Overall": 2,
        "explanation": "This response demonstrates a partial understanding of the text. Based on the premise (that the last paragraph was written to let the reader know changes made and what happened), this writer presents specific quotes and paraphrased ideas from the text. Implied information (the things around her were now different and not as they were; the little girl misses the old things and the way they were; the girl is going to consider the change; she will overcome that and move on) is used as support in this score level 2 response.",
        "essay": "The author includes the last paragraph as he does, to let the reader know of a change. The story was about a new place and changes made. The last paragraph was used to show how China's became, and what happened. 'Not like the kind we had before'. The gardens story hats to make a big change in her life. The things around her were now different, and not as they were. 'But where was the fine combo? The hibiscus hatch? The well? Her gentle grandmother?'. Things have changed from what they were before. The little girl missed the all things and the way they were.\n The author uses the last paragraph to show that the girl is going to consider the change. That nothing is the same anymore, but she will overcome that and move on."
    },
    {
        "Overall": 1,
        "explanation": "Although this response appears to be developed, the 3 reasons provided show only a minimal understanding. A relevant concept (little girl fail the test and she want to retake it is discussed but details and examples indicate a misreading of the text and question.",
        "essay": "I think three reasons why the author finish with this paragraph. First, I think, because a little girl failed the test, and she wanted to retake it. For example, I think she failed that class, but she won't fail this time so she won't to do it again, so her mother can be happy. Second, her mother, maybe is angry with her just because she rail that class, so she won to make she's mom happy. For example, maybe if She pass this test she would be able to ask her mom for something like a bike, computer, car, and shoes that third, that test it would be important for her career so if she passed, maybe she can get scholarships and don't pay nothing for college.",
    },
    {
        "Overall": 3,
        "explanation": "This response demonstrates an understanding of the complexities of the text and question. Insightful observations are interwoven with textual quotes and 'meticulous' details concerning Saeng's memories to help support the basic theme (although it may be gradual, eventually she will find the sense of comfort and belonging she once had). Although this approach is atypical of other higher level responses, it is perfectly acceptable and clearly meets the requirements of a score level 3.",
        "essay": "When reading, winter hibiscus, one learn the story behind a girl who prior to her move to the Americas, live in Asia. Although she is forced to leave behind her home, she carries on all traditions and memories that her struggle to fully adapt becomes apparent when visiting a green room which sells plants. her love for plants is revealed when she begings naming each plant, after ovserving each metialous detail. It seems as if each plan corresponds with the memory from her old home. Seang, the girl, picks up the Jasmine flower, 'she close[s] her eyes and breath[es] in deeply. The familiar fragrance filled her lungs, and Saeng could almost feel the light strands of her grandmother's long gray hair.' Saeng and her family, newcomers to America, lack money. However, Saeng who is overwhelmed with memories decided to buy expensive white hibiscus flowers anyways. She explains this as, 'I failed the test.' However, the story concludes with a girl vowing to take the test again. Firza, she decides and believes she has the ability to overcome her struggles and adapt to the new world. She is in. She realizes that, although it may be gradual, eventually, she will find the sense of comfort and belonging she once had."
    },
    {
        "Overall": 2,
        "explanation": "This response demonstrates a partial understanding of the text. Based on the premise (that the last paragraph was written to let the reader know changes made and what happened), this writer presents specific quotes and paraphrased ideas from the text. Implied information (the things around her were now different and not as they were; the little girl misses the old things and the way they were; the girl is going to consider the change; she will overcome that and move on) is used as support in this score level 2 response.",
        "essay": "The author concludes starting with this paragraph because it gives meaning to the moral of the story.\n In the story is a girl has failed a test and is consulted by a flower from her country. The girl is phased with disappointment, and must overcome this obstacle. In the last paragraph, she vows to try again as soon as the hibiscus comes back. This shows the ability to overcome obstacles.\n The moral of the story is to overcome obstacles the way that you can, so you can move on. When the girl fails a test, she is disappointed, but by means of the memories, the hibiscus planned in habits, she can find a way to overcome her disappointment.\nThe ending paragraph brings together and finalizes the idea of the girl overcoming her obstacle that is represented in the Storey 'Winter Hibiscus'."
    },
]

essay_set_4 = {
    "number_of_essays": 1772,
    "average_length": 150.0,
    "final_scores": ["domain1_score"],
    "score_ranges": [(0, 3)],
    "single_evaluator_score_ranges": [(0, 3)],
    "full_score_fn": get_final_score4,
    "prompt": prompt_essay_set4,
    "scoring_rubric": scoring_rubric_set_4,
    "examples": examples_set_4,
    "fine_grained_rubric": fine_grained_rubric_essay_set_4,
    "fine_grained_rubric_machine": fine_grained_rubric_essay_set_4_machine
}


def get_final_score5(scores1: Dict[str, int], scores2: Dict[str, int]):
    return int(scores1["Overall"])


prompt_essay_set_5 = """
Source: ```
Narciso Rodriguezfrom Home: The Blueprints of Our Lives
My parents, originally from Cuba, arrived in the United States in 1956. After living for a year in a furnished one-room apartment, twenty-one-year-old Rawedia Maria and twenty-seven-year-old Narciso Rodriguez, Sr., could afford to move into a modest, three-room apartment I would soon call home.
In 1961, I was born into this simple house, situated in a two-family, blond-brick building in the Ironbound section of Newark, New Jersey. Within its walls, my young parents created our traditional Cuban home, the very heart of which was the kitchen. My parents both shared cooking duties and unwittingly passed on to me their rich culinary skills and a love of cooking that is still with me today (and for which I am eternally grateful). Passionate Cuban music (which I adore to this day) filled the air, mixing with the aromas of the kitchen. Here, the innocence of childhood, the congregation of family and friends, and endless celebrations that encompassed both, formed the backdrop to life in our warm home.
Growing up in this environment instilled in me a great sense that “family” had nothing to do with being a blood relative. Quite the contrary, our neighborhood was made up of mostly Spanish, Cuban, and Italian immigrants at a time when overt racism was the norm and segregation prevailed in the United States. In our neighborhood, despite customs elsewhere, all of these cultures came together in great solidarity and friendship. It was a close-knit community of honest, hardworking immigrants who extended a hand to people who, while not necessarily their own kind, were clearly in need.
Our landlord and his daughter, Alegria (my babysitter and first friend), lived above us, and Alegria graced our kitchen table for meals more often than not. Also at the table were Sergio and Edelmira, my surrogate grandparents who lived in the basement apartment. (I would not know my “real” grandparents, Narciso the Elder and Consuelo, until 1970 when they were allowed to leave Cuba.) My aunts Bertha and Juanita and my cousins Arnold, Maria, and Rosemary also all lived nearby and regularly joined us at our table. Countless extended family members came and went — and there was often someone staying with us temporarily until they were able to get back on their feet. My parents always kept their arms and their door open to the many people we considered family, knowing that they would do the same for us.
My mother and father had come to this country with such courage, without any knowledge of the language or the culture. They came selflessly, as many immigrants do, to give their children a better life, even though it meant leaving behind their families, friends, and careers in the country they loved. They struggled both personally and financially, braving the harsh northern winters while yearning for their native tropics and facing cultural hardships. The barriers to work were strong and high, and my parents both had to accept that they might not be able to find the kind of jobs they deserved. In Cuba, Narciso, Sr., had worked in a laboratory and Rawedia Maria had studied chemical engineering. In the United States, they had to start their lives over entirely, taking whatever work they could find. The faith that this struggle would lead them and their children to better times drove them to endure these hard times.
I will always be grateful to my parents for their love and sacrifice. I’ve often told them that what they did was a much more courageous thing than I could have ever done. I’ve often told them of my admiration for their strength and perseverance, and I’ve thanked them repeatedly. But, in reality, there is no way to express my gratitude for the spirit of generosity impressed upon me at such an early age and the demonstration of how important family and friends are. These are two lessons that my parents did not just tell me. They showed me with their lives, and these teachings have been the basis of my life.
It was in this simple house that my parents welcomed other refugees to celebrate their arrival to this country and where I celebrated my first birthdays. It was in the warmth of the kitchen in this humble house where a Cuban feast (albeit a frugal Cuban feast) always filled the air with not just scent and music but life and love. It was here where I learned the real definition of “family.” And for this, I will never forget that house or its gracious neighborhood or the many things I learned there about how to love. I will never forget how my parents turned this simple house into a home.
— Narciso Rodriguez, Fashion designerHometown: Newark, New Jersey
“Narciso Rodriguez” by Narciso Rodriguez, from Home: The Blueprints of Our Lives. Copyright © 2006 by John Edwards.
```
Describe the mood created by the author in the memoir. Support your answer with relevant and specific information from the memoir.
"""

scoring_rubric_set_5 = {
    "Overall": {
        4: {
            "description": "The response is a clear, complete, and accurate description of the mood created by the author. The response includes relevant and specific information from the memoir.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        3: {
            "description": "The response is a mostly clear, complete, and accurate description of the mood created by the author. The response includes relevant but often general information from the memoir.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        2: {
            "description": "The response is a partial description of the mood created by the author. The response includes limited information from the memoir and may include misinterpretations.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        1: {
            "description": "The response is a minimal description of the mood created by the author. The response includes little or no information from the memoir and may include misinterpretations. OR The response relates minimally to the task.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        0: {
            "description": "The response is incorrect or irrelevant or contains insufficient information to demonstrate comprehension.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
    }
}

examples_set_5 = [
    {
        "Overall": 2,
        "explanation": "",
        "essay": "The mmood created by the author is gratful. It is a grateful mood because multiple times in this story Narisiso Rodriguez says I am so grateful for my parents, and he thanks them. One example of him saying he is grateful is paragraph 6 first line it says 'I will always be grateful to my parents for their love and sacrifice.' Another example is again in paragraph 6 it says 'I've thanked them repeatedly.' In conclusion Nariciso Rodriguez is very thankful and grateful for his parents in this story."
    },
    {
        "Overall": 0,
        "explanation": "",
        "essay": "He felt homeless when he wrote it and now the author has a home. The story has multiple moods too so it can make the story better. That is the mood in the story."
    },
    {
        "Overall": 4,
        "explanation": "",
        "essay": "In the memoir Home: The Blueprints of our lives the author creates a mood of love. He creates a loving mood because of the memories on the past. Naciso Rodriguez lived in an apartment with his parents in Newark, New Jersey. His parents did everything they could for him. For example, 'They came selflessly as many immigrants do, to give their children a better life, even though it meant leaving behind their families, friends, and careers in the country they loved.' They came from cuba. This showed love for their kids and that they would do anything to make their lives better. Another reason why the mood is love is because of what Narciso Rodriguez showed. He said 'I will always be grateful to my parents for their love and sacrifice.' That tells you that he cares about his parents. The last reason the mood is love is because 'family' was the most important thing to them. Narciso Rodriguez says 'I learned the real definition of family.' He also explains the love he had for his home and the love he learned there. in this memoir, Narciso Rodriguez shows how the mood is love by explaining what his family and home was like."
    },
    {
        "Overall": 1,
        "explanation": "",
        "essay": "The mood created by the author in the memoir is a happy mood. Narc is u loved living with parent in that apartment with all the home cooked meals. And that is the mood."
    },
    {
        "Overall": 3,
        "explanation": "",
        "essay": "The mood in this story sounds to me very thankful and happy. The boy is happy and grateful that his parents left there beloved country of cuba in order for there sun to grow up in america with a better life. He is very thankful his parents left cuba even though they had good jobs and had to start over. The mood was also very loving and emotional toward each other and there neighbors. Everyone in the neighborhood treated everyone like family and helped eachother out. In the excerpt the boy talked about all of the risks his parents took for him and how his family showed him true love. That is why I think the mood was happy, thankful, and loving."
    },
    {
        "Overall": 1,
        "explanation": "",
        "essay": "The autur has many moods in the memoir. One mood is the happiness of being a great family that supports him. Also that he loves his parents very much. Second mood that the authur has is that he he relize that he had a good birth place in newark, New Jersy. His final mood was that he had many people to hange out with. This is some of the aurthurs moods in the memoir."
    },
    {
        "Overall": 2,
        "explanation": "",
        "essay": "The mood in this memoir was thankfulness. He was thankful that his parents left Cuba for him. He was tankful of the cooking they brought him. He was thankful or their love. He was thankful for the love they gave him. He was thankful for the love the four let him. He was thankful for the home they made. He was thankful for the sacrifices his parents made. He was thankful so much that it is deffenitly the moral of the story."
    },
    {
        "Overall": 4,
        "explanation": "",
        "essay": "In the memoir by Narciso Rodriguez, he presents a very loving mood with the concepts of family, acceptance, and sacrifice. First, he talks about his 'family'. He and his family had a different definition of 'family'. Those who were in their 'family' were just those in need, it had nothing to do with he being a blood relative. Narcisco said 'My parents always kept their arms and their door open to the many people we considered family.' Second, he and his family wwere very accepting about the people around them. Narcisco lived in a very diverse neighborhood, with people from Cuba, Spain, and Italy. Unlike other places, where segregation was prominent, 'all of these cultures came together in great solidarity and friendship.' Lastly, his parents sacrificed a lot 'to give their children a better life'. They left behind their 'families, friends, and carreers in the country they loved.', enven though the money in America would be less and the jobs would be hard. All of these things that his family did, lent themselves to a loving mood, because they did it all because they loved."
    },
    {
        "Overall": 1,
        "explanation": "",
        "essay": "The author was very greatful that he was cuban and that his grandparents were cuban; and that they raised great children which are his parents. Hes happy his parents went to America and gave him a achance to get a good education The mood that was created by the author in the memoir was a positive good mood. He was just letting us know about his life as well as his family."
    },
    {
        "Overall": 2,
        "explanation": "",
        "essay": "The author in the memoir creates a mood in which everything in their house was wonderful. The author said that the aromas from the kitchen mixed with the sound of passionate luban music. Also, the author tells that friends, family, and neighbors all got together as one giant family all looking out for one another. The author also said that his parents would take in any one needing to get back on their foot because they would receive the same treatment in return. Their are many things the author of this memoir uses to creat a wonderful mood."
    },
    {
        "Overall": 3,
        "explanation": "",
        "essay": "The author was trying to get a specific mood created in the memoir. He was trying to make it a very happy and loving moood. '... Ilearned there about how to love.' He said that growing up in this house instilled him with a great sense that family had nothing to do with being a blood relative. Many people would come to visit his house because the people there were so kind. If you were down on money or anything you could go there and have a place to stay. 'The congregation of family and friends, and endless celebrations that encompassed both for med the backdrop to life in our warm home.' The last sentence he uses in his memoir sums up the whole mood of the memoir. 'I will never forget how my parents turned this simple house into a home."
    },
    {
        "Overall": 4,
        "explanation": "",
        "essay": "The author in the memoir creates a happy, admiring and touching mood in his story. When he describes the kitchen, the author sets a happy mood to his memoir. He says: 'My parents both shared cooking duties adn unwittingly passed on to me their rich culinary skills and a love of cooking that is still with me today. Passionate Cuban music filled the air, mixing with the aromas of the kitchen. Here, the innocence of childhood, the congregation of family and friends, and endless celbrations that encompassed both, formed the backdrop to life in our warm home.' The use of descriptive words helped create a happy, or elated mood to his story. In the memoir, the author creates an admiring mood by talking about how his parents came over to the United States. It says in the memoir: 'They came selflessly, as many immigrants do, to give their children a better life, even though it meant leaving behind their friends, families and careers in the country they love... The fait that this struggle would lead them and their children to better times drove them to endure these hard times.' Later on, the author also tells them of his admiration of their strength and perseverance. In the end, a touching mood is produced when the author describes his home. The author tells us that his home is where he learned the real definition of family, and for that, he will never forget the house or the neighborhood and how to love. He finishes off the touching mood by saying that he will never forget how his parents turned a simple house into a home. In the memoir, a happy, admiring, and touching mood is created by the author by expressing his thoughts and feelings."
    },
    {
        "Overall": 3,
        "explanation": "",
        "essay": "In 'Naraiso Rodriguez' the mood of the story is very loving and carring. this is true because in paragraph 2 it says, 'my parents both shared cooking duties and unwittingly passed on to me their rich colinary skills and a love for cooking is still with me today (and for which I am eternally grateful). Passionate cuban music (which I ador to this day)...' This quote shows the loving and thankfulness he has for his parents. Another example is in paragraph 5 when he says, 'My mother and father had come to this country with such courage without any knowledge of the language or the culture. They came selflessly, as many immagrints do, to give their children a better life, even though it meant leaving behind their families, friends and careers in the country they loved.' This shows the loving a care his parents had for him. This explains why the mood of this story is loving and caring."
    },
]

fine_grained_rubric_essay_set_5 = {
    'clarity': {
        4: {
            "description": "The response is a clear description of the mood created by the author"
        },
        3: {
            "description": "The response is a mostly clear description of the mood created by the author.",
        },
        2: {
            "description": "The response is a partial description of the mood created by the author.",
        },
        1: {
            "description": "The response is a minimal description of the mood created by the author.",
        },
        0: {
            "description": "The response is unclear to demonstrate comprehension.",
        },
        "overall_description": "The essay contains a clear description of the mood created by the author"
    },
    'completeness': {
        4: {
            "description": "The response is a complete description of the mood created by the author",
        },
        3: {
            "description": "The response is a mostly complete description of the mood created by the author.",
        },
        2: {
            "description": "The response is a partial description of the mood created by the author.",
        },
        1: {
            "description": "The response is a minimal description of the mood created by the author.",
        },
        0: {
            "description": "The response is insufficient to demonstrate comprehension.",
        },
        "overall_description": 'The essay contains a complete description of the mood created by the author'
    },
    'accurate': {
        4: {
            "description": "The response is an accurate description of the mood created by the author.",
        },
        3: {
            "description": "The response is a mostly accurate description of the mood created by the author.",
        },
        2: {
            "description": "The response is a partial description of the mood created by the author.",
        },
        1: {
            "description": "The response is a minimal description of the mood created by the author.",
        },
        0: {
            "description": "The response is incorrect to demonstrate comprehension.",
        },
        "overall_description": "The essay contains an accurate description of the mood created by the author"
    },
    'relevant': {
        4: {
            "description": "The response includes relevant and specific information from the memoir.",
        },
        3: {
            "description": "The response includes relevant but often general information from the memoir.",
        },
        2: {
            "description": "The response includes limited information from the memoir and may include misinterpretations.",
        },
        1: {
            "description": "The response includes little or no information from the memoir and may include misinterpretations. OR The response relates minimally to the task.",
        },
        0: {
            "description": "The response is irrelevant information to demonstrate comprehension.",
        },
        "overall_description": 'The essay includes relevant and specific information from the memoir.'
    }
}

fine_grained_rubric_essay_set_5_machine = {
    "Clarity": {
        "overall_description": "Evaluate whether the response clearly describes the mood created by the author"},

    "Completeness": {
        "overall_description": "Assess if the response provides a thorough and comprehensive description of the mood"},

    "Accuracy": {
        "overall_description": "Check if the response accurately reflects the mood as intended by the author in the memoir"},

    "Specificity": {
        "overall_description": "Look for the inclusion of relevant and specific information from the memoir to support the description of the mood"},

    "Relevance": {
        "overall_description": "Ensure that the response stays on topic and directly addresses the task of describing the mood"},

    "Interpretation": {
        "overall_description": "Determine if the response correctly interprets the mood without misrepresentations or misunderstandings"},

    "Detail": {
        "overall_description": "Examine the level of detail provided in the response, distinguishing between general and specific information"},
}

essay_set_5 = {
    "number_of_essays": 1805,
    "average_length": 150.0,
    "final_scores": ["domain1_score"],
    "score_ranges": [(0, 4)],
    "single_evaluator_score_ranges": [(0, 4)],
    "full_score_fn": get_final_score5,
    "prompt": prompt_essay_set_5,
    "scoring_rubric": scoring_rubric_set_5,
    "examples": examples_set_5,
    "fine_grained_rubric": fine_grained_rubric_essay_set_5,
    "fine_grained_rubric_machine": fine_grained_rubric_essay_set_5_machine
}


def get_final_score6(scores1: Dict[str, int], scores2: Dict[str, int]):
    return int(scores1["Overall"])


prompt_essay_set_6 = """
Source: ```
The Mooring Mastby Marcia Amidon Lüsted
When the Empire State Building was conceived, it was planned as the world’s tallest building, taller even than the new Chrysler Building that was being constructed at Forty-second Street and Lexington Avenue in New York. At seventy-seven stories, it was the tallest building before the Empire State began construction, and Al Smith was determined to outstrip it in height.
The architect building the Chrysler Building, however, had a trick up his sleeve. He secretly constructed a 185-foot spire inside the building, and then shocked the public and the media by hoisting it up to the top of the Chrysler Building, bringing it to a height of 1,046 feet, 46 feet taller than the originally announced height of the Empire State Building.
Al Smith realized that he was close to losing the title of world’s tallest building, and on December 11, 1929, he announced that the Empire State would now reach the height of 1,250 feet. He would add a top or a hat to the building that would be even more distinctive than any other building in the city. John Tauranac describes the plan:
[The top of the Empire State Building] would be more than ornamental, more than a spire or dome or a pyramid put there to add a desired few feet to the height of the building or to mask something as mundane as a water tank. Their top, they said, would serve a higher calling. The Empire State Building would be equipped for an age of transportation that was then only the dream of aviation pioneers.
This dream of the aviation pioneers was travel by dirigible, or zeppelin, and the Empire State Building was going to have a mooring mast at its top for docking these new airships, which would accommodate passengers on already existing transatlantic routes and new routes that were yet to come.
The Age of Dirigibles
By the 1920s, dirigibles were being hailed as the transportation of the future. Also known today as blimps, dirigibles were actually enormous steel-framed balloons, with envelopes of cotton fabric filled with hydrogen and helium to make them lighter than air. Unlike a balloon, a dirigible could be maneuvered by the use of propellers and rudders, and passengers could ride in the gondola, or enclosed compartment, under the balloon.
Dirigibles had a top speed of eighty miles per hour, and they could cruise at seventy miles per hour for thousands of miles without needing refueling. Some were as long as one thousand feet, the same length as four blocks in New York City. The one obstacle to their expanded use in New York City was the lack of a suitable landing area. Al Smith saw an opportunity for his Empire State Building: A mooring mast added to the top of the building would allow dirigibles to anchor there for several hours for refueling or service, and to let passengers off and on. Dirigibles were docked by means of an electric winch, which hauled in a line from the front of the ship and then tied it to a mast. The body of the dirigible could swing in the breeze, and yet passengers could safely get on and off the dirigible by walking down a gangplank to an open observation platform.
The architects and engineers of the Empire State Building consulted with experts, taking tours of the equipment and mooring operations at the U.S. Naval Air Station in Lakehurst, New Jersey. The navy was the leader in the research and development of dirigibles in the United States. The navy even offered its dirigible, the Los Angeles, to be used in testing the mast. The architects also met with the president of a recently formed airship transport company that planned to offer dirigible service across the Pacific Ocean.
When asked about the mooring mast, Al Smith commented:
[It’s] on the level, all right. No kidding. We’re working on the thing now. One set of engineers here in New York is trying to dope out a practical, workable arrangement and the Government people in Washington are figuring on some safe way of mooring airships to this mast.
Designing the Mast
The architects could not simply drop a mooring mast on top of the Empire State Building’s flat roof. A thousand-foot dirigible moored at the top of the building, held by a single cable tether, would add stress to the building’s frame. The stress of the dirigible’s load and the wind pressure would have to be transmitted all the way to the building’s foundation, which was nearly eleven hundred feet below. The steel frame of the Empire State Building would have to be modified and strengthened to accommodate this new situation. Over sixty thousand dollars’ worth of modifications had to be made to the building’s framework.
Rather than building a utilitarian mast without any ornamentation, the architects designed a shiny glass and chrome-nickel stainless steel tower that would be illuminated from inside, with a stepped-back design that imitated the overall shape of the building itself. The rocket-shaped mast would have four wings at its corners, of shiny aluminum, and would rise to a conical roof that would house the mooring arm. The winches and control machinery for the dirigible mooring would be housed in the base of the shaft itself, which also housed elevators and stairs to bring passengers down to the eighty-sixth floor, where baggage and ticket areas would be located.
The building would now be 102 floors, with a glassed-in observation area on the 101st floor and an open observation platform on the 102nd floor. This observation area was to double as the boarding area for dirigible passengers.
Once the architects had designed the mooring mast and made changes to the existing plans for the building’s skeleton, construction proceeded as planned. When the building had been framed to the 85th floor, the roof had to be completed before the framing for the mooring mast could take place. The mast also had a skeleton of steel and was clad in stainless steel with glass windows. Two months after the workers celebrated framing the entire building, they were back to raise an American flag again—this time at the top of the frame for the mooring mast.
The Fate of the Mast
The mooring mast of the Empire State Building was destined to never fulfill its purpose, for reasons that should have been apparent before it was ever constructed. The greatest reason was one of safety: Most dirigibles from outside of the United States used hydrogen rather than helium, and hydrogen is highly flammable. When the German dirigible Hindenburg was destroyed by fire in Lakehurst, New Jersey, on May 6, 1937, the owners of the Empire State Building realized how much worse that accident could have been if it had taken place above a densely populated area such as downtown New York.
The greatest obstacle to the successful use of the mooring mast was nature itself. The winds on top of the building were constantly shifting due to violent air currents. Even if the dirigible were tethered to the mooring mast, the back of the ship would swivel around and around the mooring mast. Dirigibles moored in open landing fields could be weighted down in the back with lead weights, but using these at the Empire State Building, where they would be dangling high above pedestrians on the street, was neither practical nor safe.
The other practical reason why dirigibles could not moor at the Empire State Building was an existing law against airships flying too low over urban areas. This law would make it illegal for a ship to ever tie up to the building or even approach the area, although two dirigibles did attempt to reach the building before the entire idea was dropped. In December 1930, the U.S. Navy dirigible Los Angeles approached the mooring mast but could not get close enough to tie up because of forceful winds. Fearing that the wind would blow the dirigible onto the sharp spires of other buildings in the area, which would puncture the dirigible’s shell, the captain could not even take his hands off the control levers. 
Two weeks later, another dirigible, the Goodyear blimp Columbia, attempted a publicity stunt where it would tie up and deliver a bundle of newspapers to the Empire State Building. Because the complete dirigible mooring equipment had never been installed, a worker atop the mooring mast would have to catch the bundle of papers on a rope dangling from the blimp. The papers were delivered in this fashion, but after this stunt the idea of using the mooring mast was shelved. In February 1931, Irving Clavan of the building’s architectural office said, “The as yet unsolved problems of mooring air ships to a fixed mast at such a height made it desirable to postpone to a later date the final installation of the landing gear.”
By the late 1930s, the idea of using the mooring mast for dirigibles and their passengers had quietly disappeared. Dirigibles, instead of becoming the transportation of the future, had given way to airplanes. The rooms in the Empire State Building that had been set aside for the ticketing and baggage of dirigible passengers were made over into the world’s highest soda fountain and tea garden for use by the sightseers who flocked to the observation decks. The highest open observation deck, intended for disembarking passengers, has never been open to the public.
“The Mooring Mast” by Marcia Amidon Lüsted, from The Empire State Building. Copyright © 2004 by Gale, a part of Cengage Learning, Inc.
```
Based on the excerpt, describe the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there. Support your answer with relevant and specific information from the excerpt.
"""

scoring_rubric_set_6 = {
    "Overall": {
        4: {
            "description": "The response is a clear, complete, and accurate description of the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there. The response includes relevant and specific information from the excerpt.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        3: {
            "description": "The response is a mostly clear, complete, and accurate description of the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there. The response includes relevant but often general information from the excerpt.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        2: {
            "description": "The response is a partial description of the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there. The response includes limited information from the excerpt and may include misinterpretations.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        1: {
            "description": "he response is a minimal description of the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there. The response includes little or no information from the excerpt and may include misinterpretations. OR The response relates minimally to the task.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        0: {
            "description": "The response is totally incorrect or irrelevant, or contains insufficient evidence to demonstrate comprehension.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
    },
}

examples_set_6 = [
    {
        "Overall": 2,
        "explanation": "",
        "essay": "The obstacles the builders faced were how the pur the mast on, nature, and how they did not install the mooring equipment. The builders could not put a mooring mast on the top. It would create stress on the building and something may happen. They had to modify the frame of the building so it could withstand the dirigible. Nature also caused problems. When the dirigible was tied down, ti woould shif around the mast, making it unsafe to dock. Towards the end of the project, the builders could not install the mooring equipment in time. Trring Clavan of the buildings architectural office said that they would install the landing gear later. They never did and the idea went away."
    },
    {
        "Overall": 4,
        "explanation": "",
        "essay": "In the excerpt from The Mooring Mast by Marcia Amidon Lusted, there are many obstacles that the builders of the Empire State Building faced during the attempt to allow dirigibles to dock there. First, architects realized that, 'A thousand-foot dirigible moored at the top of the building, held by a single cable tether, would add stress to the buildings frame.' The Empire State Building would not be stable enough to have a dirigible dock at the top. The second obstacle is that even if they were to find a stable way to moor the dirigible, the wind would blow it all over the place. There is no 'practical nor safe' way to tie down the dirigible. Lastly, most foreign dirigibles used hydrogen rather than helium which was a problem because hydrogen is highly flammable. If a dirigible were to catch fire while moored to the Empire State Building, above a densly populated are, the accident would be catastrophic. In conclusion, the idea of mooring dirigibles at the top of the Empire State Building was a bad decision from the beginning because of the many obstacles."
    },
    {
        "Overall": 0,
        "explanation": "",
        "essay": "The obstacles the Empire State building had for dirigible to dock there was for people to be transported. Also, they docked there to refuel and service and to let passengers on or off."
    },
    {
        "Overall": 3,
        "explanation": "",
        "essay": "The builders of the Empire State building, phased many obstacles in allowing the dirigibles to dock there. One of these problems was being able to keep the building stable. The author writes, 'A thousand fast dirigible mooral at the top of the building, held by a single cable tether, would add stress to the buildings frame.' They needed to be able to support the building, because if it collapsed, it would be devastating.\n Another obstacle that the builders faced was keeping the pedestrian safe. The dirigibles hould be weighted down, but the author writes, 'they would be dangling high above pedestrians on the street.' This would be very dangerous. If it were to fall, it would cause very much damage and kill many people. The builders faced this challenge, because there would be no other way to land the dirigibles."
    },
    {
        "Overall": 1,
        "explanation": "",
        "essay": "The constructors had a problem because they needed to find away to build the mast at the top of the building."
    },
    {
        "Overall": 2,
        "explanation": "",
        "essay": "Even widh this great idea of docking dirigibles to the roof of the Empire State building there came some obstacles. For example in paragraph 9 it says, 'A thousand-foot dirigible mooret at the top of the building, hold by a single cable tegher, would add stress to the building's frame.' That means that it would rip pairs of the building off because it would not be stable. Another problem would be in paragraph 14 it says, '... the back of the ship swivel around and around the mooring mast.' Which states that they need to secure the back or it is unsafe for people to board."
    },
    {
        "Overall": 4,
        "explanation": "",
        "essay": "The builders of the Empire State Building faced many obstacles in attempting to allow dirigibles to dock there. One was that hte mooring mast which needed to be constructed on the building in order for docking to be possible would cause the structure of the Empire State Building to 'have to be modified and strenghened to accomodate this new situation' (Lusted; paragraph 9). This is an obstacle that cost money and time huntle. Another obstacle the builders faced was that 'the winds on top of the building were constantly shifting due to violent air currents' (Lusted; paragraph 14). This problem was one that was never truly fixed and is a main reason why a dirigible never successfully docked at the Empire State Building. A third obstacle that the builders faced was that their was a law against dirigibles approaching low altitudes over populated areas. As stated n the passge, there was a 'las against airships flying too low over urban ares' that would make it 'illegal for an airship to ever tie up to the building' (Lusted; paragraph 14). Lastly, the builders of the Empire State Building had to face the obstacles that there was a safety issue with attempting to land enormous combustable ships over New York, a highly populated city. This is described in the passage by the fact that 'Most dirigibles from outside the United States used hydrogen rather than helium, and hydrogen is highly flammable' (Lusted; paragraph 13)."
    },
    {
        "Overall": 1,
        "explanation": "",
        "essay": "The obsticles were the stress of the dirigibleis I sad and the wind pressure would have to be transmitted all the way to the buildings foundation and something about the steel frame over 60 thousand dollars."
    },
    {
        "Overall": 3,
        "explanation": "",
        "essay": "In the excerpt from 'Mooring Mast' by Marceia Amidra Custed there are many obstacles the builders faced in attempting to allow dirigibles to dock on the Empire State Building. 'Most dirigibles from outside the United States used hydrogen rather than helium, and hydrogen is highly flammable (paragraph 13). 'This passes a huge problem because if the dirigible were to explode while docked on the mooring mast, then there would be a grave amount of casualties and iven if it didn't explode danger is imminant.' The winds on top of the building were constantly shifting due to vintrest air currents (paragraph 14). 'This would lead the back of the dirigible to keep spinning, thus must allow any passengers to board or leave the ship.' on existing law against airships flying too low over urban areas (paragraph 15). 'This would make it impossible to fly the dirigibles to the dock because 1,000 feet is too close to the ground to fly over an urban area especially."
    },
    {
        "Overall": 2,
        "explanation": "",
        "essay": "The builders faced many obstacles when trying to allow dirigibles. To dock at the Empire State building. One of the main reasons was docking would 'add stress to the building frame.' So they would have to find a way to send the stress eleven hundred feed down to the foundation. Also it is illegall for air ships to fly that low."
    },
    {
        "Overall": 1,
        "explanation": "",
        "essay": "Allowing the dirigibles to dock on the top of the Empire State Building was a big mistake. The Empire State Building wanted to show it was the best and could handle anything. The stand was very small to dock it on, heavy whds woud make it move from side to side, also the blings were very flammable also so couce cuase a very bad accident."
    },
    {
        "Overall": 3,
        "explanation": "",
        "essay": "In the excerpt from 'Mooring Mast', by Marcia Amidon Lusted, Is the builders of the Empire State building, experienced obstacles, and attempting to allow dirigibles to dock there. One of these problems was safety. Some dirigibles from outside the U.S. use hydrogen, not helium, which is highly flamable and would not be safe to have in New York City. Another problem that was experience was the winds on the top of the buildings always changed, so the dirigibles would bump into each other and possibly break. Last, there is a law stating air ships cannot fly too low over urban areas. Is this would make it illegal for a ship to dog at the building. These are reasons in the excerpt from the mooring mast The builders of the Empire State building, experienced obstacles in attempting to allow dirigibles to dock there."
    },
]

fine_grained_rubric_essay_set_6 = copy.deepcopy(fine_grained_rubric_essay_set_5)  # they are exactly the same!
fine_grained_rubric_essay_set_6_machine = copy.deepcopy(fine_grained_rubric_essay_set_5_machine)
essay_set_6 = {
    "number_of_essays": 1800,
    "average_length": 150.0,
    "final_scores": ["domain1_score"],
    "score_ranges": [(0, 4)],
    "single_evaluator_score_ranges": [(0, 4)],
    "full_score_fn": get_final_score6,
    "prompt": prompt_essay_set_6,
    "scoring_rubric": scoring_rubric_set_6,
    "examples": examples_set_6,
    "fine_grained_rubric": fine_grained_rubric_essay_set_6,
    "fine_grained_rubric_machine": fine_grained_rubric_essay_set_6_machine
}


def get_final_score7(scores1: Dict[str, int], scores2: Dict[str, int]):
    return (int(scores1["Ideas"]) + int(scores1["Organization"]) + int(scores1["Style"]) + int(
        scores1["Conventions"])) + \
        (int(scores2["Ideas"]) + int(scores2["Organization"]) + int(scores2["Style"]) + int(scores2[
                                                                                                "Conventions"]))  # * 2 for the Ideas is listed in the descriptions but the raw data ignore this multiplier


prompt_essay_set_7 = """
Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining.
Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience.
"""

scoring_rubric_set_7 = {
    "Ideas": {
        3: {
            "description": "Tells a story with ideas that are clearly focused on the topic and are thoroughly developed with specific, relevant details.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        2: {
            "description": "Tells a story with ideas that are somewhat focused on the topic and are developed with a mix of specific and/or general details.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        1: {
            "description": "Tells a story with ideas that are minimally focused on the topic and developed with limited and/or general details.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        0: {
            "description": "Ideas are not focused on the task and/or are undeveloped.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        "overall_description": "The essay shows ideas that are clearly focused on the topic and are thoroughly developed with specific, relevant details.",
    },
    "Organization": {
        3: {
            "description": "Organization and connections between ideas and/or events are clear and logically sequenced. ",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        2: {
            "description": "Organization and connections between ideas and/or events are logically sequenced.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        1: {
            "description": "Organization and connections between ideas and/or events are weak.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        0: {
            "description": "No organization evident.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        "overall_description": 'The essay shows organization and connections between ideas and/or events are clear and logically sequenced. '
    },
    "Style": {
        3: {
            "description": "Command of language, including effective and compelling word choice and varied sentence structure, clearly supports the writer's purpose and audience.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        2: {
            "description": "Adequate command of language, including effective word choice and clear sentences, supports the writer's purpose and audience.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        1: {
            "description": "Limited use of language, including lack of variety in word choice and sentences, may hinder support for the writer's purpose and audience.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        0: {
            "description": "Ineffective use of language for the writer's purpose and audience.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        "overall_description": "The essay shows command of language, including effective and compelling word choice and varied sentence structure, clearly supports the writer's purpose and audience."
    },
    "Conventions": {
        3: {
            "description": "Consistent, appropriate use of conventions of Standard English for grammar, usage, spelling, capitalization, and punctuation for the grade level.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        2: {
            "description": "Adequate use of conventions of Standard English for grammar, usage, spelling, capitalization, and punctuation for the grade level.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        1: {
            "description": "Limited use of conventions of Standard English for grammar, usage, spelling, capitalization, and punctuation for the grade level.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        0: {
            "description": "Ineffective use of conventions of Standard English for grammar, usage, spelling, capitalization, and punctuation.",
            "typical_elements": [],
            "fine_grained_rubric": "",
        },
        "overall_description": 'The essay shows consistent, appropriate use of conventions of Standard English for grammar, usage, spelling, capitalization, and punctuation for the grade level.'
    }
}

examples_set_7 = [
    {
        "Ideas": 1,
        "Organization": 2,
        "Style": 1,
        "Conventions": 2,
        "explanation": "Ideas in this brief response are minimally focused on patience and are developed with limited and general details.",
        "essay": "I was not patient when Jake would not give me back my Game Boy when I asked for it.he said he would, but he didn't really mean it. I had been waiting for the next few hours until he gave it back to me. That was when I decided to never let Jake to, take any of my stuff and with or not my premission."
    },
    {
        "Ideas": 0,
        "Organization": 1,
        "Style": 0,
        "Conventions": 0,
        "explanation": "In this brief response, patience is expressed by waiting (he had to wait 4 months).  However, the idea is undeveloped.",
        "essay": "My cousin went and find out a way to find out, and it is forwheer. He had to went 4 monts for it. So he bald for a mots so ge get it ruel."
    },
    {
        "Ideas": 2,
        "Organization": 3,
        "Style": 3,
        "Conventions": 3,
        "explanation": "This longer response contains two stories that are somewhat focused on patience and are developed with some specific details (I tripped over my humongous feet, she had to change my band-aids, give me benadryl, pry the tape out).",
        "essay": "I'm not very good at being patient, but my mom is. Even if I am mean to my brothers, she stays patient with me, or if I catch too much food, or, never mind, but my mom always wonderfully patient.\n My mom usually keeps a cool hat if I do something wrong (the key word is usually) Like if I hurt myself, can't choose anything to wear, things like that. Believe me, I've had my fair share of those actions. Once, I was running on the track at my elementary school, and I tripped over my humongous feet. I slammed into the cement and slid about 2 feet. By the time a teacher got over to me, I was bleeding and blubbering my hat off. They fixed me up, pretty good, but it was hard since both my hand, both my knees, and both my elbows were bleeding.The secretary, called my mom and told her what happened. My mom lost her hat for a moment, but in the end, she was extremely patient when she had had to change my Band-Aids, give me benedryl, and all that good stuff.\n There been plenty of times my mom has been patient with me; like when I broke the VCR player a little while ago, that I really don't know how I did it, but I managed to get a VCR tape stuck in the player. Evidently, I somehow got the tape stuck in between something in the machine. It's really complicated, so I don't want to go into it. This time, both my mom and my dad had to be patient with me. My dad stopped being patient when he had to unplug the VCR player, take it apart, and pry the tape out. By the time, all this was over, my parents hat, forgiven me, and the VCR player is still working.\n I'm really glad my mom stayed patient with me through thick and thin.if she hadn't, I'd be a wreck right now. An orphan, crawling on the streets, begging for a scrage of food, okay, that's a slight exaggeration, fine, a big one, but I'm still glad my mom stayed patient with me. That's why I love her so much."
    },
    {
        "Ideas": 2,
        "Organization": 2,
        "Style": 1,
        "Conventions": 1,
        "explanation": "This brief response has ideas that are somewhat focused on patience and includes a mix of general (I heard a lot of noises) and specific (I hid in the fall leaves) details.",
        "essay": "I can still remember wene I was pashint for the ferst time. it was a long timme a go we wene I was lost in the woods. At ferst I was vary scard and herd a lote of noyses but I stad callm and paishent. A bout an oher later I herd sumthing cuming throu the woods so I hid in the Falls leves. But it ternd out to be by mom. She touk me back to the house and I toold her about how I was pashenttly wating for her. And that was one time I was pashent."
    },
    {
        "Ideas": 1,
        "Organization": 1,
        "Style": 1,
        "Conventions": 0,
        "explanation": "In this lengthy response, the ideas are minimally focused on patience with few details; other information is irrelevant.",
        "essay": "Hellsow, My name is Phil and this is a story about my friend not being patient. It all stard two weeks a go I was baby siting by bater and my siter was at her kogfow class and my dad was sleeping. Now my batter is prity macher, and he can uslae do very thing him slot, but he is big gofe ball, he is 11 years old, he is cool some times. My siter is the bigst brat ever, she never does what she told, she does not get her way she cry's can, she louck's her shelf in her room, she tlack's like she is two, me, my boter, and my dad said we can't under stand her, she has a highe schkey vioce that sounds like rustey nails on 3,000 chake bord some times I seoop out eyes with a spoun, and cut off my eres with a kife, so don't have to see or here her. So any way I was siting I the coon whin I herd a knock of the door it was my frind Sam he is ok he whant to play foot ball I said ok asked my dad and wint in the back yared. The ball got stuck in the gaten I whint up and got it and trwo it down on so let me get gown he siad ok I was adout to get down then he rushed me off the side took to long I fell and brok my foot. Well thats my story see you next year."
    },
    {
        "Ideas": 3,
        "Organization": 3,
        "Style": 2,
        "Conventions": 2,
        "explanation": "This response tells a story with ideas that are clearly focused on patience (waiting for the concert to start) and is developed with specific, relevant details. ",
        "essay": "Patient. What does it mean to be patient? Being patient means you're willing to wait for something without complaining that you have to be understanding and to lerant. It's like when you wait in line to get into the mouse, or something like that. I know you've done it before I reumber when I was trying to be patient but, it was hard it was at a Jonas brothers Concert.\n I love the Jonas Brothers, so, of course, I had to see them life! My two friends, Madison and Crystal and I were going to them perform. We got to the Fox Theatre and we found our sets. There was a band playing and at first we thought it was them, but, after a while we relized it wasnt, it was there opening act, Roony.\n We thought they were only going to play one or two songs. So we sat down and waited. It seemed like they were going to play forever. I was trying to be patient and not start wanning, but it was hard.\n Fanily they came on stage. The Jonas Brothers were rockin out! Then I realized, even though it took a long time for them to come it was woorth waiting, and the concert was great.\n I was patient and I know I will be again. Being patient is something we all do."
    },
    {
        "Ideas": 2,
        "Organization": 2,
        "Style": 1,
        "Conventions": 1,
        "explanation": "Ideas in this response are somewhat focused on patience and include some specific details (it took a long time to get on the bus, we were taking a picture by the waterfall).",
        "essay": "A time when I was not patient was when men and my sister and mom went on a field trip. My sister is so anowing and she get on my nervers. OK! So they was taking a long time to get on the bus so I was like mmove to my sister. She was like stop!!! I was like no!! Then she got mad or what ever and she told my mom and I got in trouble for not being patience and I was mad at her and my mom. .Then we got there we played or what ever. Then we was taking picture By the waterfall my sister got mad and said let it be my turn I said no be pretience. Then she just walk away so I was like come on cause I felt bad for her.\n That shoed how both of us was not patience when such have and that was a example of when we was not patience."
    },
    {
        "Ideas": 0,
        "Organization": 2,
        "Style": 1,
        "Conventions": 2,
        "explanation": "This response mentions the topic of patience, but is written in the expository genre and the ideas are not focused on the task of narrative writing.",
        "essay": "You can be very successful if you are a patient that you can very successful, because can do better on papers like if the teacher is giving directions, don't go ahead or you will not get what to do. You have to be patient to do good on PayPal. That is why patience can contribute to success."
    },
    {
        "Ideas": 1,
        "Organization": 1,
        "Style": 2,
        "Conventions": 3,
        "explanation": "In this response, multiple examples of patience are given (waiting at the doctor's office, taking a test, waiting for a new video game, waiting in line) and each is developed with general details.",
        "essay": "When my mom got really sick, she had to go to the doctor start we waited three hours for the doctor. But, we sat down and were quiet. Me and my brother were playing my game boy, but we had the volume turned all the way down. Then we finally got out of the doctors office and went to Burger King, we were so hungry. We ate three cheeseburgers in two minutes that we finally got home at 10 PM and fell asleep.\n We are also being patient with the Meap test right now. It takes me so long to do the MEAP test. But I'm quiet I lay my head down and rest. I've been patient a lot of times in my life, and I am gonna have to be sometimes, and I am fine with that. I say this, because without patience, I'd be really annoyed, having to wait quietly.\n So now I'm being really patient by waiting for a garne to come out so I can play it. But even really patient people need a break every now and then that my mom and dad want me to rent the game first. They don't want to get it right when it comes out. So I have to wait a little longer, but I can live with that. They say I'm a very patient person. I had to wait two years for my Xbox 360, so I think I can wait for this.\n Besides, I had to wait anyways, because I'm already playing a game. I got a few weeks ago. This one time, we wanted to go on a ride in Michigan Adventure. It was called the Mad Mouse. We had to wait two hours, but it was so worth it. My mom was very scared that then we had to stay in a hotel. But there was a very big line there, too. But the room was very nice. Occasionally, we went swimming or went in the hot tub with bubbles."
    },
    {
        "Ideas": 3,
        "Organization": 3,
        "Style": 2,
        "Conventions": 1,
        "explanation": "Ideas in this longer response are clearly focused on patience (waiting for a ride to a friend's house) and thoroughly developed with specific, relevant details (I slammed the door, flipping through the same 5 channels, she drove 20 on a 40 mph road).",
        "essay": "I don't have Patients at all. I can tell you how. It was a bride sunny day the sun was bright the grass was swaying back in forest with the wind in the feld. I was on my way to Melissa's house and I was waiting for my mom to bring e because I did not fell like walking. I got into the car, but my mom was still in the I mean go figure she slow than the titance at full speed! I was watting Patiently but the clock keep ticking minute after minute. I said in my what was taking my mom so long? So since I don't have any Patients I got out of the slamed the door like I was throwing a tomatoe at my sister and stomed in side like a herd of animals! I yelled 'What is taking you so long mom?' come to find out she was talking on the phone to my Aunt Sue. I said to myself once again this is going to take forever I could have walked from Melissa's house and back five times now!\n While I was sitting on the couch waiting for my mom to get off the phone I turned on the t.V., but the corse nothin good was on because we did not have HD or cable like my other friends did. So basicly I was slipping thruogh the five same channels over and over again. I turn off the t.V. got up went back into the car turned on the radio and looked out the window STILL wating for my mom. Man was I getting mad. So I honked the horn over and over again. Then finlly my mom got off her lasy but said good bye to Aunt Sue and came to the car. But it did not go any faster she drove twenty on a forty miles per hour road so it took us almost ten minutes to get there when orignaly it really takes five minutes.\n When she finnley got to Melissa's house I ran out and Yelled....\n 'I'll call you when to pick me up!' Then she said okay and drove off. Man was I glad not to be with her after that."
    },
]

fine_grained_rubric_essay_set_7 = copy.deepcopy(scoring_rubric_set_7)

fine_grained_rubric_essay_set_7_machine = {
    "Ideas": {
        "overall_description": "Evaluate how clearly the story is focused on the topic of patience and how thoroughly it is developed with specific, relevant details"
    },
    "Organization": {
        "overall_description": "Assess the clarity and logical sequence of the organization and connections between ideas and/or events in the story"
    },
    "Style": {
        "overall_description": "Examine the command of language, including word choice and sentence structure, to determine how effectively it supports the writer's purpose and audience"
    },
    "Conventions": {
        "overall_description": "Check for the consistent and appropriate use of Standard English conventions, including grammar, usage, spelling, capitalization, and punctuation"
    }
}

essay_set_7 = {
    "number_of_essays": 1730,
    "average_length": 250.0,
    "final_scores": ["domain1_score"],
    "score_ranges": [(0, 30)],
    "single_evaluator_score_ranges": [(0, 3), (0, 3), (0, 3), (0, 3)],
    "full_score_fn": get_final_score7,
    "prompt": prompt_essay_set_7,
    "scoring_rubric": scoring_rubric_set_7,
    "examples": examples_set_7,
    "fine_grained_rubric": fine_grained_rubric_essay_set_7,
    "fine_grained_rubric_machine": fine_grained_rubric_essay_set_7_machine
}

IDEAS_AND_CONTENT = "Ideas and Content"
ORGANIZATION = "Organization"
SENTENCE_FLUENCY = "Sentence Fluency"
CONVENTIONS = "Conventions"
VOICE = "Voice"
WORD_CHOICE = "Word Choice"


def get_final_score8(scores1: Dict[str, int], scores2: Dict[str, int]):
    return (scores1[IDEAS_AND_CONTENT] + scores1[ORGANIZATION] + scores1[SENTENCE_FLUENCY] + scores1[
        CONVENTIONS] * 2) + \
        (scores2[IDEAS_AND_CONTENT] + scores2[ORGANIZATION] + scores2[SENTENCE_FLUENCY] + scores2[
            "Conventions"] * 2)


prompt_essay_set_8 = """
We all understand the benefits of laughter. For example, someone once said, “Laughter is the shortest distance between two people.” Many other people believe that laughter is an important part of any relationship. Tell a true story in which laughter was one element or part.
"""

scoring_rubric_set_8 = {
    IDEAS_AND_CONTENT: {
        6: {
            "description": "The writing is exceptionally clear, focused, and interesting. It holds the reader’s attention throughout. Main ideas stand out and are developed by strong support and rich details suitable to audience and purpose",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- clarity, focus, and control.
- main idea(s) that stand out.
- supporting, relevant, carefully selected details; when appropriate, use of resources provides strong, accurate, credible support.
- a thorough, balanced, in-depth explanation / exploration of the topic; the writing makes connections and shares insights.
- content and selected details that are well-suited to audience and purpose.
            """,
        },
        5: {
            "description": "The writing is clear, focused and interesting. It holds the reader’s attention. Main ideas stand out and are developed by supporting details suitable to audience and purpose.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- clarity, focus, and control.
- main idea(s) that stand out.
- supporting, relevant, carefully selected details; when appropriate, use of resources provides strong, accurate, credible support.
- a thorough, balanced explanation / exploration of the topic; the writing makes connections and shares insights.
- content and selected details that are well-suited to audience and purpose.
            """,
        },
        4: {
            "description": "The writing is clear and focused. The reader can easily understand the main ideas. Support is present, although it may be limited or rather general.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- an easily identifiable purpose.
- clear main idea(s).
- supporting details that are relevant, but may be overly general or limited in places; when appropriate, resources are used to provide accurate support.
- a topic that is explored / explained, although developmental details may occasionally be out of balance with the main idea(s); some connections and insights may be present.
- content and selected details that are relevant, but perhaps not consistently well-chosen for audience and purpose.
            """,
        },
        3: {
            "description": "The reader can understand the main ideas, although they may be overly broad or simplistic, and the results may not be effective. Supporting detail is often limited, insubstantial, overly general, or occasionally slightly off-topic.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- an easily identifiable purpose and main idea(s).
- predictable or overly-obvious main ideas; or points that echo observations heard elsewhere; or a close retelling of another work.
- support that is attempted, but developmental details are often limited, uneven, somewhat off-topic, predictable, or too general (e.g., a list of underdeveloped points).
- details that may not be well-grounded in credible resources; they may be based on clichés, stereotypes or questionable sources of information.
- difficulties when moving from general observations to specifics.
            """,
        },
        2: {
            "description": "Main ideas and purpose are somewhat unclear or development is attempted but minimal.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- a purpose and main idea(s) that may require extensive inferences by the reader.
- minimal development; insufficient details.
- irrelevant details that clutter the text.
- extensive repetition of detail.
            """,
        },
        1: {
            "description": "The writing lacks a central idea or purpose.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- ideas that are extremely limited or simply unclear.
- attempts at development that are minimal or nonexistent; the paper is too short to demonstrate the development of an idea.
            """,
        },
    },
    ORGANIZATION: {
        6: {
            "description": "The organization enhances the central idea(s) and its development. The order and structure are compelling and move the reader through the text easily.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- effective, perhaps creative, sequencing and paragraph breaks; the organizational structure fits the topic, and the writing is easy to follow.
- a strong, inviting beginning that draws the reader in and a strong, satisfying sense of resolution or closure.
- smooth, effective transitions among all elements (sentences, paragraphs, ideas).
- details that fit where placed.
            """,
        },
        5: {
            "description": "The organization enhances the central idea(s) and its development. The order and structure are strong and move the reader through the text.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- effective sequencing and paragraph breaks; the organizational structure fits the topic, and the writing is easy to follow.
- an inviting beginning that draws the reader in and a satisfying sense of resolution or closure.
- smooth, effective transitions among all elements (sentences, paragraphs, ideas).
- details that fit where placed.
            """,
        },
        4: {
            "description": "Organization is clear and coherent. Order and structure are present, but may seem formulaic.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- clear sequencing and paragraph breaks.
- an organization that may be predictable.
- a recognizable, developed beginning that may not be particularly inviting; a developed conclusion that may lack subtlety.
- a body that is easy to follow with details that fit where placed.
- transitions that may be stilted or formulaic.
- organization which helps the reader, despite some weaknesses.
            """,
        },
        3: {
            "description": "An attempt has been made to organize the writing; however, the overall structure is inconsistent or skeletal.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- attempts at sequencing and paragraph breaks, but the order or the relationship among ideas may occasionally be unclear.
- a beginning and an ending which, although present, are either undeveloped or too obvious (e.g., “My topic is...”; “These are all the reasons that...”).
- transitions that sometimes work. The same few transitional devices (e.g., coordinating conjunctions, numbering, etc.) may be overused.
- a structure that is skeletal or too rigid.
- placement of details that may not always be effective.
- organization which lapses in some places, but helps the reader in others.
            """,
        },
        2: {
            "description": "The writing lacks a clear organizational structure. An occasional organizational device is discernible; however, the writing is either difficult to follow and the reader has to reread substantial portions, or the piece is simply too short to demonstrate organizational skills.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- some attempts at sequencing, but the order or the relationship among ideas is frequently unclear; a lack of paragraph breaks.
- a missing or extremely undeveloped beginning, body, and/or ending.
- a lack of transitions, or when present, ineffective or overused.
- a lack of an effective organizational structure.
- details that seem to be randomly placed, leaving the reader frequently confused.
            """,
        },
        1: {
            "description": "The writing lacks coherence; organization seems haphazard and disjointed. Even after rereading, the reader remains confused.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- a lack of effective sequencing and paragraph breaks.
- a failure to provide an identifiable beginning, body and/or ending.
- a lack of transitions.
- pacing that is consistently awkward; the reader feels either mired down in trivia or rushed along too rapidly.
- a lack of organization which ultimately obscures or distorts the main point.
            """,
        },
    },
    VOICE: {
        6: {
            "description": "The writer has chosen a voice appropriate for the topic, purpose, and audience. The writer demonstrates deep commitment to the topic, and there is an exceptional sense of “writing to be read.” The writing is expressive, engaging, or sincere.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- an effective level of closeness to or distance from the audience (e.g., a narrative should have a strong personal voice, while an expository piece may require extensive use of outside resources and a more academic voice; nevertheless, both should be engaging, lively, or interesting. Technical writing may require greater distance.).
- an exceptionally strong sense of audience; the writer seems to be aware of the reader and of how to communicate the message most effectively. The reader may discern the writer behind the words and feel a sense of interaction.
- a sense that the topic has come to life; when appropriate, the writing may show originality, liveliness, honesty, conviction, excitement, humor, or suspense.
            """,
        },
        5: {
            "description": "The writer has chosen a voice appropriate for the topic, purpose, and audience. The writer demonstrates commitment to the topic, and there is a sense of “writing to be read.” The writing is expressive, engaging, or sincere.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by an appropriate level of closeness to or distance from the audience (e.g., a narrative should have a strong personal voice, while an expository piece may require extensive use of outside resources and a more academic voice; nevertheless, both should be engaging, lively, or interesting. Technical writing may require greater distance.).
- a strong sense of audience; the writer seems to be aware of the reader and of how to communicate the message most effectively. The reader may discern the writer behind the words and feel a sense of interaction.
- a sense that the topic has come to life; when appropriate, the writing may show originality, liveliness, honesty, conviction, excitement, humor, or suspense.
            """,
        },
        4: {
            "description": "A voice is present. The writer seems committed to the topic, and there may be a sense of “writing to be read.” In places, the writing is expressive, engaging, or sincere.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- a suitable level of closeness to or distance from the audience.
- a sense of audience; the writer seems to be aware of the reader but has not consistently employed an appropriate voice. The reader may glimpse the writer behind the words and feel a sense of interaction in places.
- liveliness, sincerity, or humor when appropriate; however, at times the writing may be either inappropriately casual or personal, or inappropriately formal and stiff.
            """,
        },
        3: {
            "description": "The writer’s commitment to the topic seems inconsistent. A sense of the writer may emerge at times; however, the voice is either inappropriately personal or inappropriately impersonal.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- a limited sense of audience; the writer’s awareness of the reader is unclear.
- an occasional sense of the writer behind the words; however, the voice may shift or disappear a line or two later and the writing become somewhat mechanical.
- a limited ability to shift to a more objective voice when necessary.
- text that is too short to demonstrate a consistent and appropriate voice.
            """,
        },
        2: {
            "description": "The writing provides little sense of involvement or commitment. There is no evidence that the writer has chosen a suitable voice.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- little engagement of the writer; the writing tends to be largely flat, lifeless, stiff, or mechanical.
- a voice that is likely to be overly informal and personal.
- a lack of audience awareness; there is little sense of “writing to be read.”
- little or no hint of the writer behind the words. There is rarely a sense of interaction between reader and writer.
            """,
        },
        1: {
            "description": "The writing seems to lack a sense of involvement or commitment.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- jno engagement of the writer; the writing is flat and lifeless.
- a lack of audience awareness; there is no sense of “writing to be read.”
- no hint of the writer behind the words. There is no sense of interaction between writer and reader; the writing does not involve or engage the reader.
            """,
        },
    },
    WORD_CHOICE: {
        6: {
            "description": "Words convey the intended message in an exceptionally interesting, precise, and natural way appropriate to audience and purpose. The writer employs a rich, broad range of words which have been carefully chosen and thoughtfully placed for impact.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- accurate, strong, specific words; powerful words energize the writing.
- fresh, original expression; slang, if used, seems purposeful and is effective.
- vocabulary that is striking and varied, but that is natural and not overdone.
- ordinary words used in an unusual way.
- words that evoke strong images; figurative language may be used.
            """,
        },
        5: {
            "description": "Words convey the intended message in an interesting, precise, and natural way appropriate to audience and purpose. The writer employs a broad range of words which have been carefully chosen and thoughtfully placed for impact.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- accurate, specific words; word choices energize the writing.
- fresh, vivid expression; slang, if used, seems purposeful and is effective.
- vocabulary that may be striking and varied, but that is natural and not overdone.
- ordinary words used in an unusual way.
- words that evoke clear images; figurative language may be used.
            """,
        },
        4: {
            "description": "Words effectively convey the intended message. The writer employs a variety of words that are functional and appropriate to audience and purpose.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- words that work but do not particularly energize the writing.
- expression that is functional; however, slang, if used, does not seem purposeful and is not particularly effective.
- attempts at colorful language that may occasionally seem overdone.
- occasional overuse of technical language or jargon.
- rare experiments with language; however, the writing may have some fine moments and generally avoids clichés.
            """,
        },
        3: {
            "description": "Language lacks precision and variety, or may be inappropriate to audience and purpose in places. The writer does not employ a variety of words, producing a sort of “generic” paper filled with familiar words and phrases.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- words that work, but that rarely capture the reader’s interest.
- expression that seems mundane and general; slang, if used, does not seem purposeful and is not effective.
- attempts at colorful language that seem overdone or forced.
- words that are accurate for the most part, although misused words may occasionally appear; technical language or jargon may be overused or inappropriately used.
- reliance on clichés and overused expressions.
- text that is too short to demonstrate variety.
            """,
        },
        2: {
            "description": "Language is monotonous and/or misused, detracting from the meaning and impact.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- words that are colorless, flat or imprecise.
- monotonous repetition or overwhelming reliance on worn expressions that repeatedly detract from the message.
- images that are fuzzy or absent altogether.
            """,
        },
        1: {
            "description": "The writing shows an extremely limited vocabulary or is so filled with misuses of words that the meaning is obscured. Only the most general kind of message is communicated because of vague or imprecise language.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- general, vague words that fail to communicate.
- an extremely limited range of words.
- words that simply do not fit the text; they seem imprecise, inadequate, or just plain wrong.
            """,
        },
    },
    SENTENCE_FLUENCY: {
        6: {
            "description": "The writing has an effective flow and rhythm. Sentences show a high degree of craftsmanship, with consistently strong and varied structure that makes expressive oral reading easy and enjoyable.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- a natural, fluent sound; it glides along with one sentence flowing effortlessly into the next.
- extensive variation in sentence structure, length, and beginnings that add interest to the text.
- sentence structure that enhances meaning by drawing attention to key ideas or reinforcing relationships among ideas.
- varied sentence patterns that create an effective combination of power and grace.
- strong control over sentence structure; fragments, if used at all, work well.
- stylistic control; dialogue, if used, sounds natural.
            """,
        },
        5: {
            "description": "The writing has an easy flow and rhythm. Sentences are carefully crafted, with strong and varied structure that makes expressive oral reading easy and enjoyable.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- a natural, fluent sound; it glides along with one sentence flowing into the next.
- variation in sentence structure, length, and beginnings that add interest to the text.
- sentence structure that enhances meaning.
- control over sentence structure; fragments, if used at all, work well.
- stylistic control; dialogue, if used, sounds natural.
            """,
        },
        4: {
            "description": "The writing flows; however, connections between phrases or sentences may be less than fluid. Sentence patterns are somewhat varied, contributing to ease in oral reading.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- a natural sound; the reader can move easily through the piece, although it may lack a certain rhythm and grace.
- some repeated patterns of sentence structure, length, and beginnings that may detract somewhat from overall impact.
- strong control over simple sentence structures, but variable control over more complex sentences; fragments, if present, are usually effective.
- occasional lapses in stylistic control; dialogue, if used, sounds natural for the most part, but may at times sound stilted or unnatural.
            """,
        },
        3: {
            "description": "The writing tends to be mechanical rather than fluid. Occasional awkward constructions may force the reader to slow down or reread.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- some passages that invite fluid oral reading; however, others do not.
- some variety in sentence structure, length, and beginnings, although the writer falls into repetitive sentence patterns.
- good control over simple sentence structures, but little control over more complex sentences; fragments, if present, may not be effective.
- sentences which, although functional, lack energy.
- lapses in stylistic control; dialogue, if used, may sound stilted or unnatural.
- text that is too short to demonstrate variety and control.
            """,
        },
        2: {
            "description": "The writing tends to be either choppy or rambling. Awkward constructions often force the reader to slow down or reread. ",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- significant portions of the text that are difficult to follow or read aloud.
- sentence patterns that are monotonous (e.g., subject-verb or subject-verb-object).
- a significant number of awkward, choppy, or rambling constructions.
            """,
        },
        1: {
            "description": "The writing is difficult to follow or to read aloud. Sentences tend to be incomplete, rambling, or very awkward.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- text that does not invite—and may not even permit—smooth oral reading.
- confusing word order that is often jarring and irregular.
- sentence structure that frequently obscures meaning.
- sentences that are disjointed, confusing, or rambling.
            """,
        },
    },
    CONVENTIONS: {
        6: {
            "description": "The writing demonstrates exceptionally strong control of standard writing conventions (e.g., punctuation, spelling, capitalization, grammar and usage) and uses them effectively to enhance communication. Errors are so few and so minor that the reader can easily skim right over them unless specifically searching for them.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- strong control of conventions; manipulation of conventions may occur for stylistic effect.
- strong, effective use of punctuation that guides the reader through the text.
- correct spelling, even of more difficult words.
- correct grammar and usage that contribute to clarity and style.
- skill in using a wide range of conventions in a sufficiently long and complex piece.
- little or no need for editing.
            """,
        },
        5: {
            "description": "The writing demonstrates strong control of standard writing conventions (e.g., punctuation, spelling, capitalization, grammar and usage) and uses them effectively to enhance communication. Errors are few and minor. Conventions support readability.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- strong control of conventions.
- effective use of punctuation that guides the reader through the text.
- correct spelling, even of more difficult words.
- correct capitalization; errors, if any, are minor.
- correct grammar and usage that contribute to clarity and style.
- skill in using a wide range of conventions in a sufficiently long and complex piece.
- little need for editing.
            """,
        },
        4: {
            "description": "The writing demonstrates control of standard writing conventions (e.g., punctuation, spelling, capitalization, grammar and usage). Significant errors do not occur frequently. Minor errors, while perhaps noticeable, do not impede readability.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- control over conventions used, although a wide range is not demonstrated.
- correct end-of-sentence punctuation; internal punctuation may sometimes be incorrect.
- spelling that is usually correct, especially on common words.
- correct capitalization; errors, if any, are minor.
- occasional lapses in correct grammar and usage; problems are not severe enough to distort meaning or confuse the reader.
- moderate need for editing.
            """,
        },
        3: {
            "description": "The writing demonstrates limited control of standard writing conventions (e.g., punctuation, spelling, capitalization, grammar and usage). Errors begin to impede readability.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- some control over basic conventions; the text may be too simple or too short to reveal mastery.
- end-of-sentence punctuation that is usually correct; however, internal punctuation contains frequent errors.
- spelling errors that distract the reader; misspelling of common words occurs.
- capitalization errors.
- errors in grammar and usage that do not block meaning but do distract the reader.
- significant need for editing.
            """,
        },
        2: {
            "description": "The writing demonstrates little control of standard writing conventions. Frequent, significant errors impede readability.",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- little control over basic conventions.
- many end-of-sentence punctuation errors; internal punctuation contains frequent errors.
- spelling errors that frequently distract the reader; misspelling of common words often occurs.
- capitalization that is inconsistent or often incorrect.
- errors in grammar and usage that interfere with readability and meaning.
- substantial need for editing.
            """,
        },
        1: {
            "description": "Numerous errors in usage, spelling, capitalization, and punctuation repeatedly distract the reader and make the text difficult to read. In fact, the severity and frequency of errors are so overwhelming that the reader finds it difficult to focus on the message and must reread for meaning. ",
            "typical_elements": [],
            "fine_grained_rubric": """
The writing is characterized by
- very limited skill in using conventions.
- basic punctuation (including end-of-sentence punctuation) that tends to be omitted, haphazard, or incorrect.
- frequent spelling errors that significantly impair readability.
- capitalization that appears to be random.
- a need for extensive editing.
            """,
        },
    }
}

examples_set_8 = [
    {
        IDEAS_AND_CONTENT: 4,
        ORGANIZATION: 4,
        VOICE: 5,
        WORD_CHOICE: 4,
        SENTENCE_FLUENCY: 4,
        CONVENTIONS: 4,
        "explanation": "Ideas and Content: 4 The writing is clear and focused, and the reader can easily understand the main idea about running for tenth grade student body president. The writing is characterized by supporting details that are relevant but they are not particularly insightful.\n Organization: 4 Organization is clear and coherent. Order and structure are present but are somewhat formulaic. Clear sequencing and paragraph breaks help the reader. The beginning starts out with a general statement and introduces the friend who runs for tenth grade student body president, and the conclusion, like the introduction, refers to the quote from the prompt. The body is easy to follow with details that fit where placed.\n Voice: 5 The writer has chosen a voice appropriate for the topic, and there is a sense of “writing to be read.” The writer seems to be aware of the reader, and the reader can discern the writer behind the words. The writing is expressive and sincere as it retells how Damon ran for office and won by sixteen votes.\n Word Choice: 4 Words effectively convey the intended message. The writer employs a variety of words that are functional and appropriate to audience and purpose, e.g., “endless responsibilities,” “expertise,” and “dirty politics.”\n Sentence Fluency: 4 The writing flows. Sentence patterns are somewhat varied, contributing to ease in oral reading. The writing is characterized by some repeated patterns of sentence structure, length, and beginnings that detract somewhat from overall impact.\n Conventions: 4 The writing demonstrates control of standard writing conventions. The writing is characterized by correct end-of-sentence punctuation; internal punctuation is sometimes incorrect. Spelling is usually correct but errors begin to distract the reader. Occasional lapses in correct grammar and usage are not severe enough to distort meaning or confuse the reader. Overall, the conventions errors do not impede readability.",
        "essay": "Running to be any member of the student council is a big job and takes a lot of courage. Dad the biggest task to take on would be president. There are so many things to keep track off and endless responsibilities. My friend Damon decided to run for 10th grade student buddy president, after talking about it all his life. Winning the election would definitely be his 'place in the sun'.\n Some problems when running for student body president is the opposing competitors. You have to be strong when it comes to criticism and competition. Unfortunately, this was not one of Damon's expertise. Almost always last when it came to dirty politics. A girl named, Sarah said, two days before the big vote, 'Damon accomplish what he promised and he still wets the bed!' People still haven't let that down and being the sensitive guy that Damon is, he wanted to be taken out from the running. \n Another hard aspect would be the student body. There are several hundred people watching your every move and judging everything you say. During this whole process, some immature kids took a marker and wrote 'Loser' all over Damon's posters another blister along the way was during third period. Some kids, we believe that it was Sarah and her crew, went to his locker and taped up a sign, saying, 'Your going down.' That wasn't too much of a disappointment for him because no one saw it. It just goes to show that you cannot be liked by everyone, but that didn't stop him. The big day was here and we were both so pumped up. The president and vice president from the previous year tabulated the votes. We knew it was going to be close. They were going to make their announcement during seventh period. Finally, with 10 minutes left in seven. The former president came over the entercom. I was holding my breath as they said, 'There has been a tie.' That was definitely a little bump in the road.\n In the end, Damon won by 14 votes!!! We were also excited for him. I took him out for ice cream, and the movie, my treat. So I guess you can accomplish anything if you wanted bad enough. Becoming president was Damon spot in the sun, because all those blisters."
    },
    {
        IDEAS_AND_CONTENT: 3,
        ORGANIZATION: 3,
        VOICE: 4,
        WORD_CHOICE: 3,
        SENTENCE_FLUENCY: 3,
        CONVENTIONS: 2,
        "explanation": "Ideas and Content: 3 The reader can understand the main ideas, but they are simplistic and the result is not effective. Supporting detail is often limited and overly general. The writing is characterized by difficulties moving from general observations about going to school to get a job to specifics about how to reach that goal.\n Organization: 3 An attempt has been made to organize the writing; however, the overall structure is skeletal. The writing is sequenced chronologically and paragraph breaks are attempted. The beginning is undeveloped and the conclusion doesn’t really complement the beginning. Transitions are used do help the reader through the piece.\n Voice: 4 A voice is present. The writer seems committed to the topic, and in places, the writing is sincere. The reader can glimpse the writer behind the words.\n Word Choice: 3 Language lacks precision and variety, producing a sort of “generic” paper filled with familiar words and phrases. The words work and are accurate but not purposeful.\n Sentence Fluency: 3 The writing tends to be mechanical rather than fluid. Occasional awkward constructions force the reader to slow down or reread (“Since I started high school, this is my secont year.”). The writing is characterized by good control over simple sentence structures but little control over more complex sentences.\n Conventions: 2 The writing demonstrates little control of standard writing conventions. The writing is characterized by many end-of-sentence punctuation errors, frequent internal punctuation errors, spelling errors that frequently distract the reader, incorrect capitalization, and errors in grammar and usage (“You got to start somewere.”). Frequent, significant errors impede readability.",
        "essay": "A job makes you PAY\n A job makes you PAY.\n When I say that I mean. When you go out in the world don't expect to walk write threw it.\n It will cause you phisical or mental Blisters. you are going to halft to work and get blisters to move up in life. Althrough my school years I was a goof off. Since I Started high school, this is my secont year. I had to work A little it harder on my grammar and spelling. All the things I was supppost to learn about in elementry and middle school. I Am doing All right and all my grades are passing. I am learning so much more. Since I half ben paying Attention. I Plan on moving up from here. You got to StArt some were.\n After high school I Am going to place myself right out there under the sun And join the MARINES. After I get my college credits and get my four years in law inforcment. SomE day be a POlice officer I Always wanted to be and Always will be.\n I t is going to tAke A log of sunnlight and a lot of blisters but I hope to live my dream as a Police officer."
    },
    {
        IDEAS_AND_CONTENT: 5,
        ORGANIZATION: 5,
        VOICE: 6,
        WORD_CHOICE: 5,
        SENTENCE_FLUENCY: 5,
        CONVENTIONS: 5,
        "explanation": "Ideas and Content: 5 The writing is clear, focused and interesting, and it holds the reader’s attention. The writing is characterized by supporting, relevant, and carefully selected details about Jordan’s attempt “to get his place in the sun” by jumping off a bridge into the water below. The writing makes connections and shares insights when Jordan’s motivation to fit in results in his death, leaving “blisters” on his parents and friends.\n Organization: 5 The order and structure are strong and move the reader through the text. The writing is characterized by effective chronological sequencing and paragraph breaks. An inviting beginning draws the reader in with its insights about being accepted in life, and the resolution is satisfying as it ties in with the beginning. Details fit where placed, and effective transitions exist among all elements.\n Voice: 6 The writer has chosen a voice appropriate for the topic, purpose, and audience. The writer demonstrates deep commitment to the topic, and there is an exceptional sense of “writing to be read.” The writer seems to be aware of the reader and of how to communicate the message most effectively.\n Word Choice: 5 Words convey the intended message in an interesting, precise, and natural way appropriate to audience and purpose. The writing is characterized by vocabulary that is natural and not overdone and words that evoke clear images, e.g., “infinite silence,” “pleading and shaking my head,” and “white gargantuan cantilever bridge.”\n Sentence Fluency: 5 The writing has an easy flow and rhythm. The writing is characterized by a natural, fluent sound and sentence structure that enhances meaning (“Then he jumped.”). The dialogue sounds natural (“Here I go!”).\n Conventions: 5 The writing demonstrates strong control of standard writing conventions and uses them effectively to enhance communication. The writing is characterized by effective use of punctuation that guides the reader through the text, e.g. dialogue. Correct spelling, capitalization, grammar and usage all contribute to clarity and style. Conventions support readability.",
        "essay": "All people, no matter what age, one to be accepted in life. Sometimes, however, people think illogically, and make bad decisions that to be accepted is to have a place in the sun among your pierce, and some people will do whatever it takes to get there.When my friend Jordan tried to get his plate in the sun, he expected to get blisters, but he didn't expect to lose his life.\n Last river is a recreation spot out in the middle of nowhere that my friend Jordan and I were on a day trip with a group of other boys. Jordan was really trying to fit in. I couldn't really understand what his motivation was or why fitting in these boys was so importanI just wanted to have some fun with my friend, but I might as well have been on the moon. We were sitting at the picnic table when Martin, the oldest of the boys, spoke.\n 'How about we go to the top of the bridge?'\n I laughed, but Jordan jumped right up and said 'Sure, let's go.' I was quite astonished, seeing as Jordan hates to swim, and the only reason went there was to jump off into the water.\n I was screaming at them, saying this was a stupid idea, but Jordan would do anything to be part of the crowd. As we approach the white gagantvan cantilever bridge I started to shiver. The bridge must have been over 100 feet high. I was trembling as we walked out to the center of the bridge. It was illegal to do this so I was trying to get them to stop.\n 'Who is going first?' asked Martin. The other boys turned and looked at Jordan. 'Jordan, you go first.'\n 'I don't know, it looks pretty high' Was Jordan's response. I looked at him, pleading and shaking my hats, trying to get him to turn back! 'GO! GO! GO! GO!' chanted the other boys. 'Jordan, lets just go back!' I screamed. 'Shut up!' Jordan gave into their pressure and cautiously climbed over the rail. I tried to grab him to pull him down. I was just too weak. The other boys held me back. 'Here I go!' said Jordan.\n Then he jumped.\n We all gasped and ran over to the site to see. There was an infinite silence as we watched his body fall further and further. When we saw the splash, the boys started cheering. I guess Jordan got what he wanted. My eyes were glued to the water. 10 seconds, 30 seconds, five minutes, we were frozen in fear and unable to see Jordan. Martin ran for help as I sang that I knew I wouldn't see him again.\n It took an hour for the ambulance to get there, and his body was not recovered for another four. They said his neck was broken on impact. Why did he do it? Why didn't I stop him?\n To this day, I ask myself these questions. Why was being accepted so important to him? I asked myself these questions. Every time I visited his grave. JORDAN THOMAS MARCH 15, 1987 - JULY 30, 2002. John wasn't the only one left with blisters after he tried to get his plates into sun that John's parents and friends, and I got blisters too. If it takes all of this to have a place in the sun, I will take rain any day."
    },
    {
        IDEAS_AND_CONTENT: 5,
        ORGANIZATION: 4,
        VOICE: 5,
        WORD_CHOICE: 5,
        SENTENCE_FLUENCY: 5,
        CONVENTIONS: 5,
        "explanation": "Ideas and Content: 5 The writing is clear, focused and interesting. Main ideas stand out and are developed by supporting details suitable to audience and purpose. The writing is characterized by an exploration of the topic that makes connections and shares insights about a freshman versus senior track experience, especially about whether the freshman would act the same as the senior due to jealousy.\n Organization: 4 Organization is clear and coherent. Order and structure are presentwithclearsequencingandparagraphbreaks. Thewritingischaracterized by a recognizable, developed beginning that is not particularly inviting, and a developed conclusion that repeats the quote from the prompt which is almost too obvious. The body is easy to follow, and overall the organization helps the reader despite some weaknesses.\n Voice: 5 The writer demonstrates commitment to the topic, and there is a sense of “writing to be read.” The writer seems to be aware of the reader, and the reader can discern the writer behind the words.\n Word Choice: 5 Words convey the intended message in an interesting, precise, and natural way appropriate to audience and purpose. The writing is characterized by accurate, specific words that energize the writing, e.g., “subdued jeering,” “wary respect,” and “amateur freshman.”\n Sentence Fluency: 5 Sentences are carefully crafted with strong and varied structure that makes expressive oral reading easy and enjoyable. Variation in sentence structure, length, and beginnings add interest to the text (“Determined not to give in” and “What bothers me most is my curiosity.”). Effective dialogue could have contributed to a 6 in this trait.\n Conventions: 5 The writing demonstrates strong control of standard writing conventions. Correct use of punctuation, spelling, capitalization, and grammar and usage support readability and leave little need for editing. A wider range of conventions attempted could raise this score to a 6.",
        "essay": "A place in the Sun\n 'If you want a place in the Sun, you will have to accept some blisters.'\n I am a varsity jumper for my high school track team and I am proud.However, accomplishing this turned out to be much harder than I anticipated that not only was it difficult because of the physical demands of the sport, but also from the stress and pressure put on by those around me.\n It is not just a rumor that seniors can be intimidating that even the sophomores and juniors acted superior, Glad to be over the freshman hurdle at last. What can make them really upset is when some silly fresh cats into their time under the spotlight.\n Positions for the varsity and junior varsity track teams are supposed to be choosing based entirely off of skill. The seniors, however, failed to see why they work for three long years just to have an amateur freshman take their spot.\n There are three varsity openings for each event, and I took one for high jump, forcing a senior down to the junior varsity level that needless to say, I was not by any means her favorite person. Consistently during our competitions, I'd hear her whispering and subdued from the sidelines. I knew what they were trying to do. She and her friends wanted me to mess up so someone more deserving would take my place.\n Determined not to give in, I continued to try my best and in turn, continued to win. Eventually, I learned to sort of wary respect from them but I won't forget the way they acted.\n What bothers me most is my curiosity. Would I have acted the same way? Would I have tried to ruin someone's chance at a place in the sun because of my jealousy? I'd like to say that I wouldn't, but I do not know for sure. If this situation is ever reverse, I will try my best not let my envy drive me to be so disheartening.\n From this experience, I learned that the quote 'If you want a place in the sun, you will have to accept some blisters', really is true. I have my chance to shine, and it wasn't easy. I hope that I can teach others about what it feels like to get blisters, so everyone's place in the sun can be enjoyable."
    },
    {
        IDEAS_AND_CONTENT: 4,
        ORGANIZATION: 4,
        VOICE: 4,
        WORD_CHOICE: 3,
        SENTENCE_FLUENCY: 3,
        CONVENTIONS: 3,
        "explanation": "Ideas and Content: 4 The writing is clear and focused, and the reader can easily understand the main idea. Support is present although it is limited. The writer tells just enough about missing Homecoming to save money to buy a truck to score a low 4. Explaining the reasons for having to pay for everything but food would have provided a more interesting and balanced paper.\n Organization: 4 Organization is clear and coherent, and order and structure are present. The sequencing is clear and paragraph breaks are correct. The beginning and ending are recognizable but not particularly inviting. Overall, the organization is almost skeletal which accounts for the low 4 score.\n Voice: 4 The writer seems committed to the topic, and there is a sense of “writing to be read.” In places, the writing is expressive and sincere (“Painful as this is, its worth it” and “I couldn’t be happier because of my new truck.”).\n Word Choice: 3 Language lacks precision and variety. The writer does not employ a variety of words, producing a sort of “generic” paper filled with familiar words and phrases. Slang is not effective (“sucked”). Words are accurate and work but rarely capture the reader’s interest.\n Sentence Fluency: 3 The writing tends to be mechanical rather than fluid. The writing is characterized by some variety in sentence structure but little variety in length. Sentences are functional but lack energy.\n Conventions: 3 The writing demonstrates limited control of standard writing conventions. End-of-sentence punctuation is usually correct, but internal punctuation contains frequent errors. Spelling errors distract the reader, and misspellings of common words occur, e.g., “somone,” “they’v,” and “anyways.” Most of the contractions are missing apostrophes. A significant need for editing exists.",
        "essay": "'If you want a place in the sun, you will have to expect some blisters'. This could mean so many things. It reminds me of a decision I had to make not to long ago. If I wanted a new truck, I had to sit out Homecoming.\n I just started working for my parents in the land scaping company they own. Ever since then, I have been stuck paying for everything myself. Except for the food that I eat. But anyways, all cloths, dances, parties and travel I have to pay for, painful as this is, its worth it.\n About a month before Homecoming. My mom told me something thet really sucked. If I wanted my truck, then I was going to have to pay for it, And of course I wanted it. Its a Diesel Ford F250 longbed.\n So I started cutling down on all of my little things a normally do. like going to the mall and going out to eat. I was worried thet someone else might buy the truck if I didnt hurry up. Thats when I decided to stay home from Homecoming. I wouldnt be able to affort it.\n This was avery hard dicision for me to make. but its only a few blisters. They'v all already healed. And I couldn't be happier because of my new truck."
    },
]

IDEAS_AND_CONTENT = "Ideas and Content"
ORGANIZATION = "Organization"
SENTENCE_FLUENCY = "Sentence Fluency"
CONVENTIONS = "Conventions"
VOICE = "Voice"
WORD_CHOICE = "Word Choice"

fine_grained_rubric_essay_set_8 = copy.deepcopy(scoring_rubric_set_8)
fine_grained_rubric_essay_set_8[IDEAS_AND_CONTENT][
    "overall_description"] = "The writing is clear, focused, and interesting. Main ideas stand out and are developed by strong support and rich details suitable to audience and purpose"
fine_grained_rubric_essay_set_8[ORGANIZATION][
    "overall_description"] = 'The organization enhances the central idea(s) and its development. The order and structure are compelling and move the reader through the text easily.'
fine_grained_rubric_essay_set_8[SENTENCE_FLUENCY][
    "overall_description"] = 'The writing has an effective flow and rhythm. Sentences show a high degree of craftsmanship, with consistently strong and varied structure that makes expressive oral reading easy and enjoyable.'
fine_grained_rubric_essay_set_8[CONVENTIONS][
    "overall_description"] = 'The writing demonstrates exceptionally strong control of standard writing conventions (e.g., punctuation, spelling, capitalization, grammar and usage) and uses them effectively to enhance communication. Errors are so few and so minor that the reader can easily skim right over them unless specifically searching for them.'
fine_grained_rubric_essay_set_8[VOICE][
    "overall_description"] = 'The writer has chosen a voice appropriate for the topic, purpose, and audience. The writer demonstrates deep commitment to the topic, and there is an exceptional sense of “writing to be read.” The writing is expressive, engaging, or sincere.'
fine_grained_rubric_essay_set_8[WORD_CHOICE][
    "overall_description"] = "Words convey the intended message in an exceptionally interesting, precise, and natural way appropriate to audience and purpose. The writer employs a rich, broad range of words which have been carefully chosen and thoughtfully placed for impact."

fine_grained_rubric_essay_set_8_machine = {
    "Ideas and Content": {
        "overall_description": "Evaluate the clarity, focus, and depth of the main ideas and supporting details, ensuring they are well-developed and suitable for the audience and purpose"
    },

    "Organization": {
        "overall_description": "Assess the effectiveness of the structure, including the sequencing of ideas, paragraph breaks, transitions, and the presence of a strong beginning and conclusion"
    },

    "Voice": {
        "overall_description": "Determine the appropriateness and engagement of the writer's voice, ensuring it suits the topic, purpose, and audience, and conveys a sense of commitment and interaction"
    },

    "Word Choice": {
        "overall_description": "Examine the precision, variety, and impact of the vocabulary used, ensuring words are carefully chosen to convey the intended message effectively and vividly"
    },

    "Sentence Fluency": {
        "overall_description": "Evaluate the flow and rhythm of the writing, looking for varied and well-crafted sentence structures that enhance readability and meaning"
    },

    "Conventions": {
        "overall_description": "Check for control over standard writing conventions such as punctuation, spelling, capitalization, grammar, and usage, ensuring they enhance readability and communication"
    },
}

essay_set_8 = {
    "number_of_essays": 918,
    "average_length": 650.0,
    "final_scores": ["domain1_score"],
    "score_ranges": [(0, 60)],
    "single_evaluator_score_ranges": [(0, 6), (0, 6), (0, 6), (0, 6), (0, 6), (0, 6)],
    "full_score_fn": get_final_score8,
    "prompt": prompt_essay_set_8,
    "scoring_rubric": scoring_rubric_set_8,
    "examples": examples_set_8,
    "fine_grained_rubric": fine_grained_rubric_essay_set_8,
    "fine_grained_rubric_machine": fine_grained_rubric_essay_set_8_machine
}

essay_set_descriptions = [essay_set_1, essay_set_2, essay_set_3, essay_set_4, essay_set_5, essay_set_6, essay_set_7,
                          essay_set_8]