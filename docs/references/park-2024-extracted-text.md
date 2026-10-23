# Park et al. (2024) - Extracted reference text

> This file is a local text extraction of the source paper. Use the accompanying PDF as the authoritative reference; OCR/table extraction may contain spacing and formatting errors.

PublishedasaconferencepaperatICLR2024
| SELF-SUPERVISED |           |     |     | CONTRASTIVE |     |     | LEARNING |     |     |     |
| --------------- | --------- | --- | --- | ----------- | --- | --- | -------- | --- | --- | --- |
|                 | LONG-TERM |     |     | FORECASTING |     |     |          |     |     |     |
FOR
JunwooPark,DaehoonGwak,JaegulChoo,EdwardChoi
KimJaechulGraduateSchoolofAI,KAIST,Daejeon,RepublicofKorea
{junwoo.park,daehoon.gwak,jchoo,edwardchoi}@kaist.ac.kr
ABSTRACT
4202 raM 42  ]GL.sc[  2v32020.2042:viXra
|     | Long-term  | forecasting |     | presents unique | challenges |     | due to the time | and memory |          |     |
| --- | ---------- | ----------- | --- | --------------- | ---------- | --- | --------------- | ---------- | -------- | --- |
|     | complexity | of handling |     | long sequences. | Existing   |     | methods, which  | rely       | on slid- |     |
ingwindowstoprocesslongsequences,struggletoeffectivelycapturelong-term
|     | variations   | that are                                                      | partially | caught | within the | short | window (i.e., | outer-window |     |     |
| --- | ------------ | ------------------------------------------------------------- | --------- | ------ | ---------- | ----- | ------------- | ------------ | --- | --- |
|     | variations). | Inthispaper,weintroduceanovelapproachthatovercomesthislimita- |           |        |            |       |               |              |     |     |
tionbyemployingcontrastivelearningandenhanceddecompositionarchitecture,
|     | specificallydesignedtofocusonlong-termvariations. |     |     |     |     |     | Tothisend,ourcontrastive |     |     |     |
| --- | ------------------------------------------------- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- |
lossincorporatesglobalautocorrelationheldinthewholetimeseries,whichfacili-
tatestheconstructionofpositiveandnegativepairsinaself-supervisedmanner.
Whencombinedwithourdecompositionnetworks,ourcontrastivelearningsig-
|     | nificantly                                         | improves | long-term    | forecasting | performance. |                                  | Extensive               | experiments |     |     |
| --- | -------------------------------------------------- | -------- | ------------ | ----------- | ------------ | -------------------------------- | ----------------------- | ----------- | --- | --- |
|     | demonstrate                                        | that     | our approach | outperforms |              | 14 baseline                      | models                  | in multiple | ex- |     |
|     | perimentsoverninelong-termbenchmarks,              |          |              |             |              | especiallyinchallengingscenarios |                         |             |     |     |
|     | thatrequireasignificantlylongoutputforforecasting. |          |              |             |              |                                  | Sourcecodeisavailableat |             |     |     |
https://github.com/junwoopark92/Self-Supervised-Contrastive-Forecsating.
1 INTRODUCTION
Time-seriesdatapresentsauniquechallengeduetoitspotentiallyinfinitelengthaccumulatingover
time, making it infeasible to process them all at once (Ding et al., 2015; Hyndman et al., 2015;
Rakthanmanonetal.,2013). Thisrequiresdifferentstrategiescomparedtoothersequencedatasuch
asnaturallanguagesentences. Toaddressthis,theslidingwindowapproach(Kohzadietal.,1996)is
commonlyemployedtopartitionasingletime-seriesdataintoshortersub-sequences(i.e.,windows)
Typically,intime-seriesforecasting,theslidingwindowapproachenablesmodelstonotonlyprocess
thelong-timeseriesbutalsocapturelocaldependenciesbetweenthepastandfuturesequencewithin
thewindows,resultinginaccurateshort-termpredictions.
Recently,asthedemandsintheindustrytopredict
more distant future increases (Ahmad et al., 2014; Variations on entire series (ETTh2)
|                                               |     |     |     |     |     | 1.00 |     |     |                    | 35000 |
| --------------------------------------------- | --- | --- | --- | --- | --- | ---- | --- | --- | ------------------ | ----- |
| Vlahogiannietal.,2014;Zhouetal.,2021),various |     |     |     |     |     |      |     |     | Autocorretion      |       |
|                                               |     |     |     |     |     | 0.75 |     |     | Fourier Components |       |
studieshavegraduallyincreasedthewindowlength. Within window 30000
|                   |     |        |      |                  |     | noitalerrocotuA 0.50 |     |     | Outer window |     |
| ----------------- | --- | ------ | ---- | ---------------- | --- | -------------------- | --- | --- | ------------ | --- |
| Transformer-based |     | models | have | reduced computa- |     |                      |     |     |              |     |
25000 edutilpmA
| tionalcostsofusinglongwindowsthroughimprove- |     |     |     |     |     | 0.25 |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- |
20000
| mentsintheattentionmechanism(Zhouetal.,2021; |     |     |     |     |     | 0.00 |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- |
15000
| Wuetal.,2021;Liuetal.,2022a).              |     |     |     | Also,CNN-based |     | 0.25 |     |     |     |       |
| ------------------------------------------ | --- | --- | --- | -------------- | --- | ---- | --- | --- | --- | ----- |
| models(Baietal.,2018;Yueetal.,2022)haveap- |     |     |     |                |     | 0.50 |     |     |     | 10000 |
plied a dilation in convolution operations to learn 0.75 5000
moredistantdependencieswhilebenefitingfromtheir
|     |     |     |     |     |     | 1.00 |     |     |     | 0   |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- |
efficientcomputationalcost. Despitetheremarkable 0 2000 4000 6000 8000 10000 12000 14000
Lag / Period
|     |     |     |     |     |     | Figure1: | Long-termvariationsspanbeyond |     |     |     |
| --- | --- | --- | --- | --- | --- | -------- | ----------------------------- | --- | --- | --- |
progressmadebythesemodels,theireffectivenessin
|                                       |     |     |     |          |     | theconventionalwindow. |       | Therearenon-zero |        |       |
| ------------------------------------- | --- | --- | --- | -------- | --- | ---------------------- | ----- | ---------------- | ------ | ----- |
| long-termforecastingremainsuncertain. |     |     |     | Sincethe |     |                        |       |                  |        |       |
|                                       |     |     |     |          |     | correlations           | (Left | Y axis) with     | longer | lags, |
extendedwindowisstillshorterthanthetotaltime
serieslength,thesemodelsmaynotlearnthelonger andFouriercomponents(RightYaxis)with
temporalpatternsthanthewindowlength. longerperiodsthanthewindowsize.
1

PublishedasaconferencepaperatICLR2024
𝑤 𝑤 𝑤
1 Yearly-long periodicity 2 3
𝑰nput 𝑶utput
𝑆𝑖𝑚(𝑤 ,𝑤 ) 𝑆𝑖𝑚(𝑤 ,𝑤 )
2 1 2 3
Figure2: (Top)Electricitytimeseriesincludingalong-termvariationbeyondwindowsize. (Bottom)
Plotted representation similarities of four models between an anchor window W and all other
2
windowsincludingW andW . Toclearlyhighlightlong-termcorrelation,wesmoothedfluctuations
1 3
causedbyshort-termcorrelation. ThedetailsofthevisualizationarefoundinAppendixC.1. Even
thoughW haveasimilartemporalpatternwithW duetoyearly-longperiodicity,threemodels,
2 1
exceptforOurs,failtolearnthisperiodicityastherepresentation. Thethreemodelsresultinnearly
identicalcosinesimilarityscores(i.e.,Sim(W ,W )≈Sim(W ,W ))betweentworepresentations
2 1 2 3
ofinputpartswithineachwindow. Thiscontributestoourmodelshowinglowermeansquarederrors
(0.275)inlong-termpredictionsthanPatchTST(0.332)andTimesNet(0.417).
Inthispaper,wefirstanalyzethelimitationsofexistingmodelstrainedwithsub-sequences(i.e.,based
onslidingwindows)forlong-termforecastingtasks. Weobservedthatmosttimeseriesoftencontain
long-termvariationswithperiodslongerthanconventionalwindowlengthsasshowninFigure1and
Figure5. Ifamodelsuccessfullycapturestheselong-termvariations,weexpecttherepresentations
oftwodistantyetcorrelatedwindowstobemoresimilarthanuncorrelatedones. However,since
thepreviousstudiesalltreateachwindowindependentlyduringtraining,itischallengingforthe
modeltocapturesuchlong-termvariationsacrossdistinctwindows. Explicitly,Figure2showsthat
therepresentationsofexistingmodelsfailtoreflectthelong-termcorrelationsbetweentwodistant
windows. Still,recentmethodstendtooverlooklong-termvariationsbyfocusingmoreonlearning
short-term variations within the window. For example, existing models based on decomposition
approaches(Zengetal.,2023;Wangetal.,2023)oftentreatthelong-termvariationspartiallycaught
in the window as simple non-periodic trends and employ a linear model to extend the past trend
into the prediction. Besides, window-unit normalization methods (Kim et al., 2021; Zeng et al.,
2023)canhinderlong-termpredictionbynormalizingnumericallysignificantvalues(e.g.,maximum,
minimum,domain-specificvaluesinthepast)thatmayhavealong-termimpactonthetimeseries.
Sincethesenormalizationmethodsareessentialformitigatingdistributionshiftproblems(Kimetal.,
2021)causedbynonstationarity(Liuetal.,2022b),anewapproachisnecessarytolearnlong-term
variationswhilestillkeepingthenormalizationmethods.
Therefore,weproposeanovelcontrastivelearningtohelpthemodelcapturelong-termdependencies
thatexistacrossdifferentwindows. Ourmethodbuildsonthefactthatamini-batchcanconsistof
windowsthataretemporallyfarapart. Itallowstheintervalbetweenwindowstospantheentire
serieslength,whichismuchlongerthanthewindowlength. Section3.1describesthedetailsofour
contrastiveloss. Moreover,weuseourcontrastivelossincombinationwithadecomposition-based
modelarchitecture,whichconsiststwobranches,namelyashort-termbranchandalong-termbranch.
Naturally,ourlossisappliedtothelong-termbranch. However,aspointedoutearlier,thelong-term
branchintheexistingdecompositionarchitecturehasbeencomposedofasinglelinearlayer,which
isunsuitableforlearninglong-termrepresentations. Thus,asexplainedinSection3.2,weredesign
thedecompositionarchitecturewherethelong-termbranchhassufficientcapacitytolearnlong-term
representationfromourloss. Insummary,themaincontributionsofourworkareasfollows:
• Ourfindingsrevealthatthelong-termperformancesofexistingmodelsarepoorasthose
modelsoverlookedthelong-termvariationsbeyondthewindow.
• WeproposeAutoCon,anovelcontrastivelossfunctiontolearnalong-termrepresentationby
constructingpositiveandnegativepairsacrossdistantwindowsinaself-supervisedmanner.
• Extensiveexperimentsonninedatasetsdemonstratethattheproposeddecompositionarchi-
tecturetrainedwithAutoConachievesperformanceimprovementsofupto34%compared
toatotalof14concurrentmodelsincludingthreerepresentationmethods.
2

PublishedasaconferencepaperatICLR2024
2 RELATED WORK
ContrastiveLearningforTime-seriesForecasting Contrastivelearning(Chenetal.,2020;Khosla
etal.,2020;Zhaetal.,2022)isatypeofself-supervisedlearningtechniquethathelpsmodelslearn
usefulrepresentationsofdatawithouttheneedforexplicitlabelingofdata. Motivatedbytherecent
successofcontrastivelearningincomputervision,numerousmethods(Tonekabonietal.,2021;Yue
etal.,2022;Wooetal.,2022a)havebeenproposedintime-seriesanalysis. Incontrastivelearning,
sincehowtoconstructpositivepairshasagreatinfluenceontheperformance,theymainlyproposed
positive pair construction strategies such as temporal consistency (Tonekaboni et al., 2021), sub-
seriesconsistency(Franceschietal.,2019),andcontextualconsistency(Yueetal.,2022). However,
these strategies have a limitation in that only temporally close samples are selected as positives,
overlooking the periodicity in the time series. Due to the periodicity, there may be more similar
negativesamplesthanpositivelyselectedsamples. Recently,CoST(Wooetal.,2022a)triedtolearn
arepresentationconsideringperiodicitythroughFrequencyDomainContrastiveloss,butitcouldnot
considerperiodicitybeyondthewindowlengthbecauseitstillusesaugmentationforthewindow. In
thetime-serieslearningframework,wefocusonthefactthatrandomlysampledsequencesinabatch
canbefarfromeachotherintime. Therefore,weproposeanovelselectionstrategytochoosenot
onlylocalpositivepairsbutalsoglobalpositivepairsbetweenthewindowsinthebatch.
Decomposition-basedModelsforLong-termForecasting Time-seriesdecomposition(Cleveland
et al., 1990) is a well-established technique that involves breaking down a time series into its
individualcomponents,suchastrend,seasonal,andremaindercomponents. Bydecomposingatime
seriesintothesecomponents, itbecomeseasiertoanalyzeeachcomponent’sbehaviorandmake
moreinterpretablepredictions. Thus, decomposition-basedmodels(Wuetal.,2021;Zhouetal.,
2022b;Wangetal.,2023)havegainedpopularityintime-seriesforecasting,astheyofferrobustand
interpretablepredictions,evenwhentrainedoncomplextimeseries. Recently,DLinear(Zengetal.,
2023)hasdemonstratedexceptionalperformancebyusingadecompositionblockandasinglelinear
layerforeachtrendandseasonalcomponent. However,ouranalysisindicatesthattheselinearmodels
areeffectiveincapturinghigh-frequencycomponentsthatimpactshort-termpredictions,whilethey
oftenmisslow-frequencycomponentsthatsignificantlyaffectlong-termpredictions. Therefore,a
singlelinearmodelmaybesufficientforshort-termprediction,butitisinadequateforlong-term
prediction. In light of this limitation, we propose a model architecture that includes layers with
varyingcapacitiestoaccountfortheuniquepropertiesofbothcomponents.
3 METHOD
Notations Wefirstdescribetheforecastingtaskwiththeslidingwindowapproach(Zhouetal.,
2021;Wuetal.,2021;Parketal.,2023),whichcoversallpossiblein-outputsequencepairsofthe
entire time series S = {s ,...,s } where T denotes the length of the observed time series and
1 T
s ∈Rcisobservationwithcdimension. Forthesimplicityinexplainingourmethodology,wesetthe
t
dimensioncto1throughoutthispaper. ByslidingawindowwithafixedlengthW onS,weobtain
thewindowsD ={W }M whereW =(X ,Y )aredividedintotwoparts: inputsequenceX =
t t=1 t t t t
{s ,...,s }withtheinputlengthI,andoutputsequenceY ={s ,...,s }withthe
t t+I−1 t t+I t+I+O−1
outputlengthOtopredict. Also,wedenotetheglobalindexsequenceofW asT ={t+i}W−1.
t t i=0
3.1 AUTOCORRELATION-BASEDCONTRASTIVELOSSFORLONG-TERMFORECASTING
Missing Long-term Dependency in the Window Many real-world time series exhibit diverse
long-term and short-term variations (Wu et al., 2021; 2023; Wang et al., 2023). In such cases, a
forecastingmodelmaystruggletopredictlong-termvariations,asthesevariationsarenotcaptured
withinthewindow. Wefirstidentifytheselong-termvariationsusingautocorrelation,inspiredbythe
stochasticprocesstheory(Chatfield&Xing,2019;Papoulis&UnnikrishnaPillai,2002). Forareal
discrete-timeprocess{S },wecanobtaintheautocorrelationfunctionR (h)usingthefollowing
t SS
equation: T
1 (cid:88)
R (h)= lim S S (1)
SS T→∞T t t−h
t=1
Theautocorrelationmeasuresthecorrelationbetweenobservationsatdifferenttimes(i.e.,timelagh).
Astrongcorrelationcloseto1or-1indicatesthatallpointsseparatedbyhintheseriesS arelinearly
related,movinginthesameoroppositedirectionforpositiveornegativesigns,respectively. Inother
words, autocorrelationcanbeutilizedtoforecastfuturevariationsthatarehintervalawaybased
oncurrentvariations. Althoughrecentmethodshaveleveragedautocorrelationtodiscoverperiod-
3

PublishedasaconferencepaperatICLR2024
Value
|     |     |     |     |     | --- Long-term variations |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | --- | --- | --- |
Time
| 𝓡   | 𝓢𝓢 𝒕 | 𝒕   |     |     |         |              |     |     | 𝒕   |     |     |
| --- | ---- | --- | --- | --- | ------- | ------------ | --- | --- | --- | --- | --- |
|     | 𝟏    |     | 𝟐   |     | 𝓡 𝓢𝓢(|𝒕 | 𝟑-𝒕 |) = 0.2 |     |     | 𝟑   |     |     |
𝟐
𝓡 𝓢𝓢(|𝒕 𝟐-𝒕 𝟏 |) = 0
|     |     |     |     |     |     |     |     |     | 𝓡 𝓢𝓢(|𝒕 𝟑-𝒕 |) = 0.5 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -------------------- | --- | --- |
𝟏
Lag
𝓡 (|𝒕 -𝒕 |) > 𝓡 (|𝒕 -𝒕 |)   𝓡 (|𝒕 -𝒕 |) > 𝓡 (|𝒕 -𝒕 |) 𝓡 (|𝒕 -𝒕 |) > 𝓡 (|𝒕 -𝒕 |)
|        | 𝓢𝓢 𝟏     | 𝟑      | 𝓢𝓢 𝟏 𝟐 |        | 𝓢𝓢 𝟐       | 𝟑       | 𝓢𝓢 𝟐 𝟏 |        | 𝓢𝓢 𝟑 𝟏  | 𝓢𝓢     | 𝟑 𝟐 |
| ------ | -------- | ------ | ------ | ------ | ---------- | ------- | ------ | ------ | ------- | ------ | --- |
|        |          |        |        |        |            | W2, W2  |        |        | W3, W1  | W3, W2 |     |
|        | : W1, W3 | W1, W2 |        |        | : : W2, W3 |         |        |        | :       |        |     |
| Anchor | Pos      |        | Neg    | Anchor |            |         | Neg    | Anchor | Pos     | Neg    |     |
Pos
Figure3: ExampleoftherelativeselectionstrategyinourAutoCon. Threewindowsaresampled
from different times t , t , and t on the entire series to make up the batch. In this batch, there
|     |     |     | 1 2 | 3   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
areatotalofthreepossiblepositivepairs(i.e.,duetothreeanchors). Eachpaircalculatesaglobal
autocorrelationwhoselagisthetimedistanceofthetwowindowsconstitutingthepair. Then,by
comparingtheautocorrelationwithotherremainingpairs,thepairswithlowerautocorrelationthan
theanchorpositivepairaredesignatedasnegativepairs.
baseddependencies(Wuetal.,2021;Wangetal.,2022), theyonlyapplyittocapturevariations
withinthewindow,overlookinglong-termvariationsthatspanbeyondthewindow. Butasshown
inFigure1,non-zerocorrelationsexistoutsidetheconventionalwindowlength. Forthefirsttime,
we propose a representation learning method via contrastive learning to capture these long-term
variationsquantifiedbytheglobalautocorrelation. Notethat,todistinguishourmethodfromthose
thatuselocalautocorrelationwithinagivenwindow,werefertotheautocorrelationcalculatedacross
theentiretimeseriesastheglobalautocorrelation.
Autocorrelation-basedContrastiveLoss(AutoCon) Wenotethatamini-batchcanconsistof
windowsthataretemporallyveryfarapart. Thistimedistancecanbeaslongastheentireseries
lengthT,whichismuchlongerthanthewindowlengthW. Basedonthisfact,weaddresslong-term
dependenciesthatexistthroughouttheentireseriesbyestablishingrelationshipsbetweenwindows.
Concretely,wedefinetherelationshipbetweenthetwowindowsbasedontheglobalautocorrelation.
AnytwowindowsW andW obtainedattwodifferenttimest andt eachhaveW observations
|                                  |     | t1  |     | t2  |      |            | 1   | 2       |               |     |     |
| -------------------------------- | --- | --- | --- | --- | ---- | ---------- | --- | ------- | ------------- | --- | --- |
|                                  |     |     |     |     |      | +i}W−1andT |     | +j}W−1. |               |     |     |
| withgloballyindexedtimesequenceT |     |     |     |     | ={t  |            |     | ={t     | Then,wedenote |     |     |
|                                  |     |     |     |     | t1 1 | i=0        | t2  | 2       | j=0           |     |     |
∈RW×W.
timedistancesbetweenallpairsoftwoobservationsineachwindowasamatrixD This
matrix D contains time distance as elements D = |(t +j)−(t +i)|. In the two windows,
|     |     |     |     |     |     | i,j | 2   | 1   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the time distances between the same phase (i.e., i = j) all have the same value |t − t |, and
|     |     |     |     |     |     |     |     |     |     | 1   | 2   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
they are represented by the diagonal terms {D }W−1 of the matrix. Therefore, based on this
|     |     |     |     |     |     | i,i i=1 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- |
representativeness,weleveragetheglobalautocorrelationR SS (|t 1 −t 2 |)todefinetherelationship
betweenthetwowindowsasfollows:
|        |                                                              |     |     | r(T | ,T )=|R | (|t | −t |)| |     |     |     | (2) |
| ------ | ------------------------------------------------------------ | --- | --- | --- | ------- | --- | ------ | --- | --- | --- | --- |
|        |                                                              |     |     |     | t1 t2   | SS  | 1 2    |     |     |     |     |
| whereR | SS denotetheglobalautocorrelationcalculatedfromtrainseriesS. |     |     |     |         |     |        |     |     |     |     |
Now,wedesignalosstoensurethatthesimilaritiesbetweenallpairsofwindowrepresentationsfollow
theglobalautocorrelationmeasuredinthedataspace. Toachievethis,wedefinepositiveandnegative
samplesinarelativemannerinspiredbySupCR(Zhaetal.,2022)forregressiontasksintheimage
domain. However,unlikeSupCRwhichusesannotatedlabelstodeterminetherelationshipbetween
images, we use the global autocorrelation R to determine the relationship between windows,
SS
makingourapproachanunsupervisedmethod. Wefeedamini-batchX ∈RN×I consistingofN
windowstotheencodertoobtainrepresentationsv ∈RN×I×dwherev =Enc(X,T). Indexedby
thewindowsi,ourautocorrelation-basedcontrastiveloss,calledAutoCon,isthencomputedoverthe
| representations{v(i)}N |         |            | withthecorrespondingtimesequence{T(i)}N |          |           |     |                      |                          | as:             |         |                 |
| ---------------------- | ------- | ---------- | --------------------------------------- | -------- | --------- | --- | -------------------- | ------------------------ | --------------- | ------- | --------------- |
|                        |         |            | i=1                                     |          |           |     |                      | i=1                      |                 |         |                 |
|                        |         | N          |                                         | N        |           |     | exp (cid:0) Sim      | (cid:0) v(i),v(j)(cid:1) | /τ              | (cid:1) |                 |
|                        |         | 1 (cid:88) | 1                                       | (cid:88) |           |     |                      |                          |                 |         |                 |
| L                      | =−      |            |                                         |          | r(i,j)log |     |                      |                          |                 |         |                 |
|                        | AutoCon | N          | N −1                                    |          | (cid:80)N | 1   |                      |                          | (cid:0) (cid:0) |         | (cid:1) (cid:1) |
|                        |         |            |                                         |          |           |     | [k̸=i,r(i,k)≤r(i,j)] | exp                      | Sim v(i),v(k)   |         | /τ              |
|                        |         | i=1        |                                         | j=1,j̸=i |           | k=1 |                      |                          |                 |         |                 |
(3)
whereSim(·,·)measuresthesimilaritybetweentworepresentations(e.g.,cosinesimilaritybetween
maxpooledv(i) alongwiththetimeaxis(Yueetal.,2022)),andr(i,j) =r(T(i),T(j))represents
4

PublishedasaconferencepaperatICLR2024
the global correlation between two windows. During training, there are a total of N ×(N −1)
possiblepairsindexedby(i,j). Eachpair(i.e.,asananchorpair)designatesitselfasarelatively
positivepairbyconsideringanypairsthatexhibittheglobalautocorrelationr(i,k) lowerthanthat
r(i,j)oftheanchorpairasnegativepairs. Figure3describessamplecasesofourselectionstrategy
in the given batch. Since a different set of windows form the batch in each iteration, we expect
that the representations reflect the global autocorrelations of all possible distances. The relative
selectionstrategydoesnotguaranteethatthepositivewindowhasahighcorrelationclosetoone;it
onlyrequiresahighercorrelationthanothernegativewindowsinthesamebatch. Consequently,we
introducer(i,j)asweightstodifferentiatebetweenpositivepairswithvaryingdegreesofcorrelation,
similartofocalloss(Linetal.,2017). TominimizeL ,theencoderlearnsrepresentationsso
AutoCon
thatthepairswithhighcorrelationarecloserthanthepairswithlowcorrelation.
OurAutoConoffersseveralnotableadvantagesoverconventionalcontrastive-basedmethods. First,
althoughAutoConisanunsupervisedrepresentationmethod,itdoesnotrelyondataaugmentation,
whichiscommoninmostcontrastive-basedapproaches(Tonekabonietal.,2021;Yueetal.,2022;
Wooetal.,2022a). Theaugmentation-basedmethodsrequireadditionalcomputationcostscaused
bytheaugmentationprocessandincreasetheforward-backwardprocessfortheaugmenteddata.
Also,existingcontrastivelearningmethodsconsideronlytemporallyclosesamplesaspositivepairs
withinwindows. Thisultimatelyfailstoappropriatelylearnrepresentationsofthewindowsthatare
distantfromeachotherbutaresimilarduetolong-termperiodicity. Consequently,ourmethodis
computationallyefficientandabletolearnlong-termrepresentations,enhancingtheabilitytopredict
long-termvariationseffectively.
3.2 DECOMPOSITIONARCHITECTUREFORLONG-TERMREPRESENTATION
𝑁= 3
𝒳norm
Linear
𝒴෠
short
𝒳seq Window 𝒳ഥ Window
𝒴gt
MSE Loss
Normalization Denormalization
𝒳norm 𝒴෠
𝒯 Encoder 𝑣1 𝑣2𝑣3 Multi-scale
MA Decoder 𝒴෠
long
GlobalAutocorrelationℛ𝒮𝒮
AutoCon
B
Fo
a
r
c
w
kw
ar
a
d
r d
p
p
at
a
h
th
L
S
o
h
n
o
g
rt
-
-
t
t
e
e
r
r
m
m
b
b
r
r
a
a
n
n
c
c
h
h
Figure4: Anoverviewoftheredesignedarchitectureforlong-termrepresentationandforecasting
Existing models commonly adopt the decomposition architecture that has a seasonal branch and
a trend branch to achieve disentangled seasonal and trend prediction. To emphasize that trends
arelong-termvariationspartiallycaughtinthewindow,weregardthetrendbranchasalong-term
branchandtheseasonalbranchasashort-termbranch. OurAutoConmethodisdesignedtolearn
long-termrepresentations,makingitnaturalnottouseitintheshort-termbranchtoenforcelong-term
dependencies. However,integratingAutoConwithcurrentdecompositionarchitecturespresentsa
challengebecausebothbranchessharethesamerepresentation(Wuetal.,2021;Zhouetal.,2022b;
Liuetal.,2022b),orthelong-termbranchconsistsofalinearlayerthatisnotsuitableforlearning
representations (Zeng et al., 2023; Wang et al., 2023). Moreover, we observe that recent linear-
basedmodels(Zengetal.,2023)outperformcomplicateddeepmodelsatshort-termpredictions,
leavingdoubtswhetheradeepmodelisnecessarytolearnthehigh-frequencyvariations. Basedon
theseconsiderations,weredesignamodelarchitecturewithwell-definedexistingblockstorespect
temporallocalityforshort-termandglobalityforlong-termforecastingasshowninFigure4. Our
decompositionarchitecturehasthreemainfeatures.
NormalizationandDenormalizationforNonstationarity First,weusewindow-unitnormalization
anddenormalizationmethods(Equation4)(Kimetal.,2021;Zengetal.,2023)asfollows:
X =X −X¯, Y =(Y +Y )+X¯ (4)
norm pred short long
whereX¯isthemeanoftheinputsequence. Thesesimplemethodshelptoeffectivelyalleviatethe
distributionshiftproblem(Kimetal.,2021)bynonstationarityofthereal-worldtimeseries.
Short-termBranchforTemporalLocality Next,weobservethatshort-periodvariationsoften
repeatmultipletimeswithintheinputsequenceandexhibitsimilarpatternswithtemporallyclose
5

PublishedasaconferencepaperatICLR2024
sequences. Thislocalityofshort-termvariationssupportstherecentsuccessoflinear-basedmod-
els(Zengetal.,2023),whichusesequentialinformationofadjacentsequencesonly. Therefore,we
employthelinearlayerfortheshort-termpredictionasfollows:
Y =Linear(X ). (5)
short norm
Long-termBranchforTemporalGlobality Thelong-termbranch,designedtoapplytheAutoCon
method,employsanencoder-decoderarchitecture. Theencoderwithsufficientcapacitytolearnthe
long-termpresentationleveragesbothsequentialinformationandglobalinformation(i.e.,timestamp-
basedfeaturesderivedfromT)asfollows:
v =Enc(X ,T). (6)
norm
Thechoiceofnetworkfortheencoderisflexibleaslongastherearenoissuesinprocessinglong
sequences. Weoptedfortemporalconvolutionnetworks(Baietal.,2018)(TCNs),widelyusedin
learningtime-seriesrepresentation(Yueetal.,2022),foritscomputationalefficiency. Thedecoder
employsthemulti-scaleMovingAverage(MA)block(Wangetal.,2023),withdifferentkernelsizes
{k }n tocapturemultipleperiodsbasedontherepresentationvasfollows:
i i=1
n
Yˆ = 1 (cid:88) AvgPool(Padding(MLP(v))) . (7)
long n ki
i=1
TheMAblockattheheadofthelong-termbranchsmoothsoutshort-termfluctuations,naturally
encouragingthebranchtofocusonlong-terminformation. Ourredesignedarchitectureisoptimized
bytheobjectivefunctionLasfollows:
L=L +λ·L . (8)
MSE AutoCon
where the mean square error (MSE) and the AutoCon loss are combined with the weight λ as
a hyperparameter. The hyperparamer sensitivity analysis is available in Appendix A.6. Detailed
descriptionsofeachoperation(e.g.,Linear,Padding,andMLP)canbefoundintheAppendixA.1.
4 EXPERIMENTS
Tovalidateourproposedmethod,weconductedextensiveexperimentsonninereal-worlddatasets
fromsixdomains:mechanicalsystems(ETT),energy(Electricity),traffic(Traffic),weather(Weather),
economics(Exchange),anddisease(ILI).Wefollowstandardprotocol(Wuetal.,2021)andsplitall
datasetsintotraining,validation,andtestsetsinchronologicalorderbytheratioof6:2:2. Weselect
thelatestbaselinemodelswithdifferentarchitecturescategorizedintolinear-based(Zhouetal.,2022a;
Zengetal.,2023),CNN-based(Wuetal.,2023;Wangetal.,2023),andTransformer-based(Zhou
etal.,2022b;Liuetal.,2022b;Nieetal.,2023). Additionally,wecomparedourmodelwithtwo
models(Challuetal.,2023;Zhang&Yan,2023)thatfocusonlearninginter-channeldependencies
formultivariateforecasting. AppendixAprovidesmoredetailedinformationaboutthedatasetsand
baselineimplementations.
4.1 MAINRESULTS
Extended Long-term Forecasting In our pursuit to better evaluate our model’s performance in
predictinglong-termvariations—whichtendtohaveincreasingsignificanceastheforecastlength
extends—wedesignedourexperimentstoextendthepredictionlengthOforeachdataset. Thisshift
fromtheconventionalbenchmarkexperiments,whichtypicallypredictupto720steps,allowsusto
explorethemodel’scapabilityinmorechallengingforecastscenarios. Forthedatasetswithlonger
totallengths,suchasETTh,Electricity,Traffic,andWeather,wetripledthepredictionlengthfrom720
to2160.Also,forExchangeandILIdatasetswithshortertotallengths,weextendtheoutputlengthup
to1080and112,respectively. Overall,Table1showsthatourmodelwithAutoConoutperformedthe
state-of-the-artbaselinesbyachievingfirstplace42timesintheunivariatesetting. Whenexamining
theperformancechangesaccordingtolength,ourmodelshowedsignificantimprovementcompared
tootherbestmodelswhenpredictingfurtherintothefuture(e.g.,onaverage,errorsdecreasedby5%
at96and720,andby12%at1440and2160). Theseresultsempiricallydemonstratethecontribution
ofourAutoConineffectivelycapturinglong-termvariationsthatexistbeyondthewindow.
DatasetAnalysis Sinceourgoalistolearnlong-termvariations,theperformanceimprovements
ofourmodelcanbeaffectedbythemagnitudeandthenumberoflong-termvariations. Figure5
showsvariousyearly-longbusinesscyclesandnaturalcyclesuniquetoeachdataset. Forinstance,
6

PublishedasaconferencepaperatICLR2024
Table1: Extendedlong-termforecastingresultswithvariouspredictionlengthsOandthebestinput
lengthI ∈{48,96,168,336}foreachmodelexceptforIllnessdatasetwithI =14. Redandblue
numbersdenotethebestandsecond-bestresults,respectively. ThefullbenchmarkswithETTh1and
ETTmareavailableinAppendixD.
|        |      | TimesNet | MICN   | PatchTST | DLinear | FiLM Nonstationary |         | FEDformer |
| ------ | ---- | -------- | ------ | -------- | ------- | ------------------ | ------- | --------- |
| Models | Ours |          |        |          |         |                    |         |           |
|        |      | (2023)   | (2023) | (2023)   | (2023)  | (2022a)            | (2022b) | (2022b)   |
O MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE
96 0.124 0.269 0.139 0.290 0.122 0.264 0.136 0.292 0.128 0.271 0.129 0.275 0.192 0.343 0.129 0.277
2hTTE 720 0.177 0.344 0.207 0.370 0.313 0.457 0.233 0.392 0.319 0.461 0.256 0.407 0.231 0.394 0.273 0.419
1440 0.176 0.340 0.192 0.358 0.520 0.599 0.351 0.481 0.514 0.597 0.389 0.506 0.211 0.379 0.384 0.487
2160 0.198 0.358 0.263 0.413 0.759 0.734 0.610 0.659 0.740 0.728 0.610 0.645 0.240 0.399 0.919 0.737
yticirtcelE 96 0.196 0.313 0.286 0.386 0.241 0.367 0.227 0.336 0.207 0.322 0.394 0.451 0.332 0.426 0.279 0.393
720 0.275 0.386 0.417 0.471 0.336 0.446 0.332 0.426 0.304 0.412 0.467 0.504 0.505 0.533 0.417 0.486
1440 0.338 0.441 0.491 0.523 0.419 0.504 0.482 0.537 0.395 0.484 0.625 0.610 0.577 0.574 0.651 0.609
2160 0.380 0.481 0.536 0.547 0.421 0.501 0.768 0.644 0.415 0.496 0.938 0.758 0.642 0.610 0.896 0.714
96 0.132 0.206 0.145 0.219 0.168 0.256 0.192 0.296 0.219 0.327 0.264 0.334 0.247 0.326 0.220 0.312
cfifarT 720 0.144 0.225 0.163 0.269 0.304 0.394 0.213 0.318 0.309 0.419 0.247 0.329 0.277 0.360 0.255 0.344
1440 0.174 0.251 0.188 0.292 0.375 0.443 0.246 0.341 0.353 0.409 0.311 0.390 0.303 0.361 0.297 0.376
2160 0.175 0.252 0.190 0.304 0.360 0.426 0.261 0.353 0.324 0.386 0.988 0.745 0.222 0.317 0.317 0.394
96 0.521 0.522 0.584 0.536 0.569 0.525 0.545 0.539 0.579 0.529 0.589 0.533 0.636 0.567 0.703 0.625
rehtaeW 720 0.963 0.715 1.090 0.753 1.080 0.754 0.987 0.752 1.007 0.706 1.003 0.728 1.007 0.725 1.114 0.822
1440 1.280 0.835 1.547 0.926 1.351 0.863 1.342 0.860 1.299 0.823 1.472 0.900 1.394 0.867 1.435 0.919
2160 1.415 0.887 1.744 0.994 1.544 0.937 1.506 0.924 1.454 0.887 1.712 0.988 1.598 0.944 1.786 1.054
egnahcxE 48 0.051 0.172 0.054 0.178 0.054 0.181 0.068 0.197 0.049 0.170 0.052 0.173 0.054 0.178 0.059 0.184
360 0.448 0.527 0.479 0.532 0.459 0.536 0.548 0.573 0.485 0.531 0.492 0.534 0.493 0.541 0.528 0.556
720 1.067 0.794 1.239 0.856 1.383 0.927 1.264 0.859 1.718 1.024 1.291 0.864 1.358 0.894 1.381 0.903
1080 1.004 0.792 1.327 0.900 4.874 1.972 1.255 0.873 4.982 1.973 1.670 1.010 1.774 1.058 1.600 0.980
14 0.725 0.574 1.414 0.735 0.815 0.701 1.558 0.965 1.397 0.901 1.079 0.739 1.107 0.698 0.773 0.619
ILI 28 0.887 0.683 1.604 0.854 1.670 1.062 1.878 1.110 2.008 1.134 1.315 0.887 1.515 0.767 0.989 0.770
56 0.807 0.725 1.021 0.787 1.757 1.210 1.451 1.028 1.584 1.075 1.080 0.891 0.895 0.742 0.856 0.741
112 1.499 1.038 1.669 1.072 3.593 1.759 2.846 1.438 3.332 1.572 2.608 1.387 1.724 1.108 1.660 1.097
| 1stCount | 42          | 0   | 2     | 0   | 4       | 0   | 0       | 0   |
| -------- | ----------- | --- | ----- | --- | ------- | --- | ------- | --- |
|          | Electricity |     | ETTh2 |     | Traffic |     | Weather |     |
60
| 6000  |     |      |     | 0.2     |     | 500     |                     |                        |
| ----- | --- | ---- | --- | ------- | --- | ------- | ------------------- | ---------------------- |
| 4000  |     | 40   |     |         |     |         |                     |                        |
| 2000  |     | 20   |     | 0.1     |     | 400     |                     |                        |
| 1.000 |     | 0    |     | 10.0.00 |     | 13.0000 | Long-term variation |                        |
|       |     | 1.00 |     |         |     |         |                     | Autocorrelation        |
| 0.75  |     | 0.75 |     | 0.75    |     | 0.75    |                     | Maximum Correlation    |
|       |     |      |     |         |     |         |                     | W it h i n  w in d o w |
| 0.50  |     | 0.50 |     | 0.50    |     | 0.50    |                     | Ou t e r  w in d o w   |
| 0.25  |     | 0.25 |     | 0.25    |     | 0.25    |                     |                        |
| 0.00  |     | 0.00 |     | 0.00    |     | 0.00    |                     |                        |
0 5000 10000 15000 20000 25000 0 2500 5000 7500 1000012500 0 5000 10000 15000 0 10000 20000 30000 40000 50000
Figure5: Theouter-windowautocorrelationexistsinvaryingdegreesinfourdatasets.
the ETTh2 and Electricity datasets have strong long-term correlations with peaks at several lags
repeatedmultipletimes. Thus,ourmethodontheETTh2andElectricitydatasetsexhibitedsignificant
performance gains, which are 34% and 11% reduced error compared to the second-best model,
respectively. Incontrast,theWeatherdatasethasrelativelylowercorrelationsoutsidethewindows
thantheaforementionedtwodatasets. Thisleadsourmodeltoshowtheleastimprovementwitha3%
reducederrorontheWeatherdataset. Asaresult,ourmethod’ssuperioritymanifestsmorestrongly
forthedatasetswithstrongerlong-termcorrelation,thusempiricallyvalidatingourcontribution.
ExtensiontoMultivariateForecasting AsshowninTable2,ourmethodisapplicableinmulti-
variateforecastingbycalculatingautocorrelationonaper-channelbasisandthenfollowingachannel
independenceapproach(Nieetal.,2023). AppendixA.2describesdetailsformultivariatesetting.
4.2 MODELANALYSIS
TemporalLocalityandGlobality AsmentionedinSection3.2,weproposedamodelarchitecture
that combines the advantages of both linear models for locality and deep models for globality.
Figure6(a)demonstratesthat,forshort-termpredictionsupto96units,thelinearmodel(DLinear)
achievedalowererrorratethanthedeepmodelssuchasTimesNet,Nonstationary,andFEDformer.
However,theerrorofDLinearstartedtodivergeasthepredictionlengthextended. Conversely,the
TimesNetandNonstationarymaintainedaconsistenterrorrateevenwiththeincreaseinprediction
lengthbutdidn’tperformaswellasthelinearmodelforshort-termpredictions. Theseobservations
servedasthemotivationforourdecompositionarchitecturethatisproficientinbothshort-termand
long-termpredictions(bluelineinFigure6(a)).
AblationStudies Here,weconductedablationstudiestovalidateeachcomponentofourmethod.
Figure 6(b) shows the results of the ablation study conducted on the full model, and when the
7

PublishedasaconferencepaperatICLR2024
|     | (a) Comparison with baselines |     | (b) Ablations in our method |                     |
| --- | ----------------------------- | --- | --------------------------- | ------------------- |
| E S |                               |     | E S                         |                     |
| M   |                               |     | M                           |                     |
|     | I/O ratio = 1                 |     | I/O ratio = 1               |                     |
|     | Prediction Length O           |     |                             | Prediction Length O |
Figure6: Comparisonofforecastingerror(MSE)accordingtopredictionlengthO. (a)comparison
withbaselinemodelsand(b)comparisonbetweenablationsofourmethodontheETTh2dataset.
Table 2: Multivariate forecasting results on ETT datasets with different prediction lengths O ∈
{96,192,336,720} and the input length I = 96. Due to lack of space, we report the averaged
| performanceoffourlengthsettings. |           | ThefullbenchmarkisavailableatAppendixD. |                 |                     |
| -------------------------------- | --------- | --------------------------------------- | --------------- | ------------------- |
|                                  | TimesNet* | MICN Crossformer                        | N-HiTS DLinear* | ETSformer* LightTS* |
Models Ours (2023) (2023) (2023) (2023) (2023) (2022b) (2022)
Dataset MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE
ETTh1 0.442 0.431 0.458 0.450 0.559 0.535 0.591 0.550 0.502 0.490 0.456 0.452 0.542 0.510 0.491 0.479
ETTh2 0.372 0.401 0.414 0.427 0.588 0.525 0.885 0.673 0.545 0.491 0.559 0.515 0.439 0.452 0.602 0.543
ETTm1 0.390 0.400 0.400 0.406 0.392 0.414 0.503 0.489 0.428 0.436 0.403 0.407 0.429 0.425 0.435 0.437
ETTm2 0.281 0.325 0.291 0.333 0.328 0.382 0.593 0.535 0.346 0.383 0.350 0.401 0.293 0.342 0.409 0.436
∗denotestheresults,whicharetakenfromTimesNet(Wuetal.,2023).
short-termbranchwasremoved,itshowedasignificanterrorforshort-termpredictions. Whenthe
long-termbranchwasremoved,itshowedasignificanterrorforlong-termpredictions. Also,without
theintegrationofourAutoCon,thelong-termperformancewasdegraded.AsdemonstratedinTable3,
thesetrendswereconsistentacrossavarietyofdatasets.
4.3 COMPARISONWITHREPRESENTATIONLEARNINGMETHODS
Figure7: ThefiguredisplaysUMAP(McInnesetal.,2018)visualizationresultsovertherepresenta-
tionsofdifferentfourmethodsontheETTh2dataset. OurAutoCondemonstratesclearcontinuityand
clusteringbetweenadjacentmonths,indicatinganunderstandingoflong-termvariation. Incontrast,
theothermodelsappeartolackthisperceptibleone-yearlong-termstructure,possiblyduetothe
limitedrepresentationlearningwithinthewindow.
Wealsodemonstratetheeffectivenessofourmethodincapturinglong-termrepresentationsbeyond
thewindowcomparedtoexistingtime-seriesrepresentationlearningmethods. TS2Vec(Yueetal.,
2022)andCoST(Wooetal.,2022a)arebothunsupervisedcontrastivelearningmethods,withTS2Vec
onlyconsideringtheaugmenteddataofthesametimeindexasapositivepairandCoSTusinga
lossthattakesintoaccountperiodicity,butbothhavethelimitationofonlybeingeffectivewithin
| Table3:  | Ablationoftheshort-term,long-termbranch,andAutoConinOurs. |       |       |       |
| -------- | --------------------------------------------------------- | ----- | ----- | ----- |
| Datasets |                                                           | ETTh1 | ETTh2 | ETTm2 |
Predictionlength 96 720 1440 2160 96 720 1440 2160 192 1440 2880 4320
Ours MSE 0.055 0.078 0.078 0.074 0.125 0.177 0.176 0.198 0.093 0.214 0.211 0.214
w/oShort-term MSE 0.071 0.093 0.126 0.094 0.204 0.271 0.263 0.302 0.186 0.313 0.300 0.288
w/oLong-term MSE 0.055 0.084 0.093 0.108 0.126 0.242 0.353 0.592 0.093 0.235 0.258 0.326
w/oAutoCon MSE 0.061 0.082 0.096 0.130 0.147 0.214 0.212 0.236 0.118 0.302 0.254 0.237
8

PublishedasaconferencepaperatICLR2024
Table4: Comparisonwithrepresentationlearningmethods.
|     | Datasets |     | ETTm1 |     |     | Exchange |     | ILI |     |
| --- | -------- | --- | ----- | --- | --- | -------- | --- | --- | --- |
Predictionlength 192 1440 2880 4320 48 360 720 1080 14 28 56 112
AutoCon MSE 0.041 0.090 0.089 0.082 0.051 0.448 1.067 1.003 0.724 0.886 0.810 1.499
LaST(2022) MSE 0.053 0.120 0.204 0.274 0.051 0.418 2.022 5.529 1.730 2.712 1.694 3.206
CoST(2022) MSE 0.059 0.130 0.199 0.192 0.054 0.451 1.703 4.470 0.746 1.114 1.490 3.155
TS2Vec(2022) MSE 0.064 0.150 0.171 0.161 0.059 0.516 1.387 5.337 2.161 2.262 3.043 4.098
a window. Therefore, while they show competitive performance in relatively short lengths, they
failtopredictaccuratelyforlong-termperiods. LaST(Wangetal.,2022)isadecomposition-based
representationlearningmethodandalsoshowscompetitiveperformanceinshort-termpredictions,
butfailstopredictaccuratelyforlong-termperiods. Figure7showsthelearnedrepresentationby
AutoConwiththeotherthreemethods. AppendixC.2providesexperimentalprotocolsandfurther
comparisonexperiments.
4.4 COMPUTATIONALEFFICIENCYCOMPARISON
Ourproposedmodelshowscompetitivecomputationalefficiencyamongotherdeepmodels. Specifi-
cally,ontheETTdataset,ourmodelwithoutAutoConexhibitscomputationaltimesof31.1ms/iter,
second best after the linear models. Even with the integration of AutoCon during training, the
computational cost does not increase significantly (33.2 ms/iter) since there is no augmentation
processandtheautocorrelationcalculationoccursonlyonceduringtheentiretraining. Consequently,
ourmodel’scomputationalefficiencysurpassesexistingTransformer-basedmodels(Nonstationary
365.7ms/iter)andrecentstate-of-the-artCNN-basedmodels(TimesNet466.1ms/iter). Detailed
comparisonscanbefoundinAppendixB.4.
| 5   | DISCUSSION | &   | LIMITATIONS |     |     |     |     |     |     |
| --- | ---------- | --- | ----------- | --- | --- | --- | --- | --- | --- |
Ourproposedmethodmitigatestheconstraintoftheslidingwindowapproachbylearningthelong-
termvariationsbeyondthewindow. Nevertheless,weexaminewhetherthelimitationofthesliding
windowwepointoutcanbesolvedbysimplyincreasingthewindowlengthwithoutourmethod,and
alsoelucidatealimitationofourmethod.
Can we use a very long window to capture long-term variations? Given a time series S of
lengthT,thenumberofwindowsM isT −(I+O)+1. ThisimpliesthatastheinputlengthI (i.e.,
datacomplexity)increases,whilekeepingtheoutputlengthOfixed,thenumberofdatainstances
(i.e.,windows)availableforlearningdecreases,potentiallymakingthemodelmoresusceptibleto
overfitting (Park et al., 2023) as shown in Figure 8. Consequently, it is challenging for input
|      | DLinear |      | TimesNet |      | MICN | Nonstationary_Transformer |       | FEDformer |     |
| ---- | ------- | ---- | -------- | ---- | ---- | ------------------------- | ----- | --------- | --- |
| 0.22 |         | 0.12 |          |      |      |                           |       |           |     |
|      |         |      |          | 0.35 |      | 0.16                      | 0.275 |           |     |
| 0.20 |         |      |          |      |      | 0.15                      |       |           |     |
|      |         | 0.10 |          | 0.30 |      |                           | 0.250 |           |     |
| 0.18 |         |      |          |      |      | 0.14                      | 0.225 |           |     |
0.25
| ESM 0.16 | train | 0.08 |     |      |     | 0.13    | 0.200     |     |     |
| -------- | ----- | ---- | --- | ---- | --- | ------- | --------- | --- | --- |
|          | test  |      |     |      |     | 0.12    |           |     |     |
| 0.14     |       | 0.06 |     | 0.20 |     |         | 0.175     |     |     |
|          | ours  |      |     |      |     | 0.11    | 0.150     |     |     |
| 0.12     |       |      |     | 0.15 |     |         |           |     |     |
|          |       | 0.04 |     |      |     | 0 . 1 0 | 0 . 1 2 5 |     |     |
| 0.10     |       |      |     | 0.10 |     | 0 . 0 9 |           |     |     |
0 . 1 0 0
| 0.08 |         | 0.02    |     | 0.05    |     | 0.08        | 0.075 |         |     |
| ---- | ------- | ------- | --- | ------- | --- | ----------- | ----- | ------- | --- |
|      | 192 480 | 960 192 | 480 | 960 192 | 480 | 960 192 480 | 960   | 192 480 | 960 |
Input Length
Figure8: Thefigureshowsthetraining(blueline)andtest(orangeline)MSElosstraceplotsoffive
modelsastheinputlengthgraduallyincreasesfrom192to960ontheETTh1withtheoutput-720
setting. Inthefigure,theredhorizontallinerepresentsourtestlosswiththeinput-96setting.
sequencestobelongenoughtocoveralllong-termvariationspresentinthedata,andmodelsoften
struggletocapturevariationsoutsidethewindow. Therefore,thelimitationwepointoutregarding
theslidingwindowapproachisvalidinmostsituationsandworthaddressing. AppendixB.1presents
comprehensiveexperimentsandempiricalfindingsobtaineduponincreasingthewindowlength.
CanAutocorrelationcapturealllong-termvariations? Whileautocorrelationservesasavaluable
tool for capturing certain long-term variations, its linearity assumption limits its effectiveness in
dealingwiththenon-linearpatternsandrelationshipsprevalentinreal-worldtimeseriesdata. By
consideringhigher-ordercorrelations,non-lineardependencies,andexternalfactors,wearelikelyto
achieveevenmoreaccurateandcomprehensivelong-termforecasting.
9

PublishedasaconferencepaperatICLR2024
ACKNOWLEDGMENTS
ThisworkwassupportedbytheNationalResearchFoundationofKorea(NRF)grantfundedbythe
Koreagovernment(MSIT)(No.NRF-2020H1D3A2A03100945,No.NRF-2022R1A2B5B02001913),
and Institute of Information & communications Technology Planning & Evaluation (IITP) grant
fundedbytheKoreagovernment(MSIT)(No.2019-0-00075,No.2022-0-00984).
REFERENCES
AhmadSAhmad,MohammadYHassan,MdPauziAbdullah,HasimahARahman,FHussin,Hayati
Abdullah,andRahmanSaidur. Areviewonapplicationsofannandsvmforbuildingelectrical
energyconsumptionforecasting. RenewableandSustainableEnergyReviews,33:102–109,2014.
ShaojieBai,JZicoKolter,andVladlenKoltun. Anempiricalevaluationofgenericconvolutionaland
recurrentnetworksforsequencemodeling. arXivpreprintarXiv:1803.01271,2018.
CristianChallu, KinGOlivares, BorisNOreshkin, FedericoGarzaRamirez, MaxMergenthaler
Canseco,andArturDubrawski. Nhits: Neuralhierarchicalinterpolationfortimeseriesforecasting.
InProc.theAAAIConferenceonArtificialIntelligence(AAAI),2023.
ChrisChatfieldandHaipengXing. Theanalysisoftimeseries: anintroductionwithR. CRCpress,
2019.
TingChen,SimonKornblith,MohammadNorouzi,andGeoffreyHinton. Asimpleframeworkfor
contrastivelearningofvisualrepresentations. InProc.theInternationalConferenceonMachine
Learning(ICML),pp.1597–1607.PMLR,2020.
RobertBCleveland,WilliamSCleveland,JeanEMcRae,andIrmaTerpenning.Stl:Aseasonal-trend
decomposition. J.Off.Stat,6(1):3–73,1990.
LarsDannecker. Energytimeseriesforecasting: efficientandaccurateforecastingofevolvingtime
seriesfromtheenergydomain. Springer,2015.
RuiDing,QiangWang,YingnongDang,QiangFu,HaidongZhang,andDongmeiZhang. Yading:
Fastclusteringoflarge-scaletimeseriesdata. ProceedingsoftheVLDBEndowment,8(5):473–484,
2015.
Jean-YvesFranceschi,AymericDieuleveut,andMartinJaggi. Unsupervisedscalablerepresentation
learningformultivariatetimeseries. 32,2019.
RobJHyndman,EaroWang,andNikolayLaptev. Large-scaleunusualtimeseriesdetection. In2015
IEEEinternationalconferenceondataminingworkshop(ICDMW),pp.1616–1619.IEEE,2015.
Muhammad Faisal Iqbal, Muhammad Zahid, Durdana Habib, and Lizy Kurian John. Efficient
prediction of network traffic for real-time applications. Journal of Computer Networks and
Communications,2019,2019.
Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron
Maschinot,CeLiu,andDilipKrishnan. Supervisedcontrastivelearning. 33:18661–18673,2020.
TaesungKim,JinheeKim,YunwonTae,CheonbokPark,Jang-HoChoi,andJaegulChoo. Reversible
instancenormalizationforaccuratetime-seriesforecastingagainstdistributionshift. InProc.the
InternationalConferenceonLearningRepresentations(ICLR),2021.
Nowrouz Kohzadi, Milton S. Boyd, Bahman Kermanshahi, and Iebeling Kaastra. A com-
parison of artificial neural network and time series models for forecasting commodity
prices. Neurocomputing, 10(2):169–181, 1996. ISSN 0925-2312. doi: https://doi.org/
10.1016/0925-2312(95)00020-8. URLhttps://www.sciencedirect.com/science/
article/pii/0925231295000208. FinancialApplications,PartI.
VincentLeGuenandNicolasThome. Shapeandtimedistortionlossfortrainingdeeptimeseries
forecastingmodels. InProc.theAdvancesinNeuralInformationProcessingSystems(NeurIPS),
pp.4191–4203,2019.
10

PublishedasaconferencepaperatICLR2024
Tsung-Yi Lin, Priya Goyal, Ross B. Girshick, Kaiming He, and Piotr Dolla´r. Focal loss for
dense object detection. 2017 IEEE International Conference on Computer Vision (ICCV), pp.
2999–3007, 2017. URL http://openaccess.thecvf.com/content_ICCV_2017/
papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf.
Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dust-
dar. Pyraformer: Low-complexitypyramidalattentionforlong-rangetimeseriesmodelingand
forecasting. InProc.theInternationalConferenceonLearningRepresentations(ICLR),2022a.
YongLiu,HaixuWu,JianminWang,andMingshengLong. Non-stationarytransformers: Rethinking
thestationarityintimeseriesforecasting. InProc.theAdvancesinNeuralInformationProcessing
Systems(NeurIPS),2022b.
LelandMcInnes,JohnHealy,andJamesMelville. Umap: Uniformmanifoldapproximationand
projectionfordimensionreduction. arXivpreprintarXiv:1802.03426,2018.
YuqiNie,NamHNguyen,PhanwadeeSinthong,andJayantKalagnanam. Atimeseriesisworth
64words: Long-termforecastingwithtransformers. InProc.theInternationalConferenceon
LearningRepresentations(ICLR),2023.
AthanasiosPapoulisandSUnnikrishnaPillai.Probability,randomvariablesandstochasticprocesses.
2002.
JunwooPark,JungsooLee,YounginCho,WoncheolShin,DongminKim,JaegulChoo,andEdward
Choi. Deepimbalancedtime-seriesforecastingvialocaldiscrepancydensity. InProc.ofMachine
Learning and Knowledge Discovery in Databases: Research Track (ECML/PKDD), pp. 139–
155. Springer Nature Switzerland, 2023. doi: 10.1007/978-3-031-43424-2 9. URL https:
//link.springer.com/content/pdf/10.1007/978-3-031-43424-2_9.
ThanawinRakthanmanon,BilsonCampana,AbdullahMueen,GustavoBatista,BrandonWestover,
QiangZhu,JesinZakaria,andEamonnKeogh. Addressingbigdatatimeseries: Miningtrillions
of time series subsequences under dynamic time warping. ACM Transactions on Knowledge
DiscoveryfromData(TKDD),7(3):1–31,2013.
HiroakiSakoeandSeibiChiba. Dynamicprogrammingalgorithmoptimizationforspokenword
recognition. IEEEtransactionsonacoustics,speech,andsignalprocessing,26(1):43–49,1978.
SanaTonekaboni,DannyEytan,andAnnaGoldenberg. Unsupervisedrepresentationlearningfor
timeserieswithtemporalneighborhoodcoding. arXivpreprintarXiv:2106.00750,2021.
Jose´ FTorres,DalilHadjout,AbderrazakSebaa,FranciscoMart´ınez-A´lvarez,andAliciaTroncoso.
Deeplearningfortimeseriesforecasting: asurvey. BigData,9(1):3–21,2021.
EleniIVlahogianni,MatthewGKarlaftis,andJohnCGolias. Short-termtrafficforecasting: Where
weareandwherewe’regoing. TransportationResearchPartC:EmergingTechnologies,43:3–19,
2014.
HuiqiangWang,JianPeng,FeihuHuang,JinceWang,JunhuiChen,andYifeiXiao. Micn: Multi-
scalelocalandglobalcontextmodelingforlong-termseriesforecasting. InProc.theInternational
ConferenceonLearningRepresentations(ICLR),2023.
ZhiyuanWang,XoveeXu,WeifengZhang,GoceTrajcevski,TingZhong,andFanZhou. Learning
latentseasonal-trendrepresentationsfortimeseriesforecasting. InProc.theAdvancesinNeural
InformationProcessingSystems(NeurIPS),2022.
GeraldWoo,ChenghaoLiu,DoyenSahoo,AkshatKumar,andStevenHoi.Cost:Contrastivelearning
ofdisentangledseasonal-trendrepresentationsfortimeseriesforecasting. 2022a.
GeraldWoo,ChenghaoLiu,DoyenSahoo,AkshatKumar,andStevenHoi. Etsformer: Exponential
smoothingtransformersfortime-seriesforecasting. arXivpreprintarXiv:2202.01381,2022b.
HaixuWu,JiehuiXu,JianminWang,andMingshengLong.Autoformer:Decompositiontransformers
withauto-correlationforlong-termseriesforecasting. InProc.theAdvancesinNeuralInformation
ProcessingSystems(NeurIPS),2021.
11

PublishedasaconferencepaperatICLR2024
Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet:
Temporal 2d-variation modeling for general time series analysis. In Proc. the International
ConferenceonLearningRepresentations(ICLR),2023.
Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, and
BixiongXu.Ts2vec:Towardsuniversalrepresentationoftimeseries.InProc.theAAAIConference
onArtificialIntelligence(AAAI),volume36,pp.8980–8987,2022.
Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series
forecasting? InProc.theAAAIConferenceonArtificialIntelligence(AAAI),2023.
KaiwenZha,PengCao,YuzheYang,andDinaKatabi. Supervisedcontrastiveregression. arXiv
preprintarXiv:2210.01189,2022.
TianpingZhang,YizhuoZhang,WeiCao,JiangBian,XiaohanYi,ShunZheng,andJianLi. Lessis
more: Fastmultivariatetimeseriesforecastingwithlightsampling-orientedmlpstructures. arXiv
preprintarXiv:2207.01186,2022.
YunhaoZhangandJunchiYan. Crossformer: Transformerutilizingcross-dimensiondependency
for multivariate time series forecasting. In Proc. the International Conference on Learning
Representations(ICLR),2023.
HaoyiZhou,ShanghangZhang,JieqiPeng,ShuaiZhang,JianxinLi,HuiXiong,andWancaiZhang.
Informer: Beyondefficienttransformerforlongsequencetime-seriesforecasting. InProc.the
AAAIConferenceonArtificialIntelligence(AAAI),2021.
Tian Zhou, Ziqing Ma, Qingsong Wen, Liang Sun, Tao Yao, Wotao Yin, Rong Jin, et al. Film:
Frequencyimprovedlegendrememorymodelforlong-termtimeseriesforecasting. InProc.the
AdvancesinNeuralInformationProcessingSystems(NeurIPS),2022a.
TianZhou,ZiqingMa,QingsongWen,XueWang,LiangSun,andRongJin. Fedformer: Frequency
enhanceddecomposedtransformerforlong-termseriesforecasting. InProc.theInternational
ConferenceonMachineLearning(ICML),2022b.
A REPRODUCIBILITY
A.1 DETAILSOFOURMETHODIMPLEMENTATION
Inthissubsection,weprovideadetailedexplanationoftheoperationsthatwereusedinSection3.2.
Firstly, Linear signifies a linear layer along the time dimension. When provided with an input
sequenceX,theoutputYˆiscomputedas:
Yˆ =W ·X (9)
time
whereX ∈RI×c,Yˆ ∈RO×c,andW ∈RO×I.
time
Next,wedescribePadding(·)andAvgpool(·)alongwiththetimeaxisasfollows:
X =Padding(X) (10)
pad
X =Avgpool(X ) (11)
avg pad k
k−1
1 (cid:88)
X [t,c]= X [t+i,c] (12)
avg k pad
i=0
whereX ∈R(I+2(k−1))×cispaddedsequenceonbothsideswithneighboringvaluestopreserve
pad
inputlengthafterapplyingAvgpool(·). ThevariouskernelsizesofAvgpool(·)areselectedtohandle
themulti-periodicities,whichareobservedinreal-worldtime-seriesdata.
MLP denotestheuseoftwolinearlayerswithanactivationfunctiong,specificallyGELU.Given
theinputsequenceX,theoutputY iscomputedas:
Yˆ =g(W ·X +b )·W +b (13)
time time channel channel
12

PublishedasaconferencepaperatICLR2024
|     | ∈Rd,b |     | ∈Rc,W |     | ∈RO×I,andW |     |     | ∈Rd×c. |     |     |
| --- | ----- | --- | ----- | --- | ---------- | --- | --- | ------ | --- | --- |
whereb
|     | time | channel |     | time |     |     | channel |     |     |     |
| --- | ---- | ------- | --- | ---- | --- | --- | ------- | --- | --- | --- |
TocomputeAutoCon,globalautocorrelationmustfirstbedetermined. Aspartofthepreprocessing
process, we calculate autocorrelation for the entire training series, excluding validation and test
series. Thisautocorrelationcomprisesbothlongandshort-periodvariations. However,inourpursuit
of disentangled long-term representation, we intend that the long-term branch address only low-
frequencyvariationswithlongperiods. Therefore,wesmoothoutshort-periodfluctuationsinthe
seriesbeforecomputingtheautocorrelation.
Our training protocol is identical to conventional training methods, except for the inclusion of
AutoConasanadditionalloss,inadditiontotheforecastingloss. Toclarifythisfurther,wealso
presentthealgorithm.
| Algorithm1AutoCon: |     |     | Autocorrelation-basedConstrastiveLearningFramework |     |     |     |     |     |     |     |
| ------------------ | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
}M
Require: EntiretimeseriesS ={s ,...,s },TrainingsetD ={(X ,Y ),T ,
|     |     |     |     | 1   | T   |     |     | t t | t t=1 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- |
AutoConweightλwhereT denotesatotallengthandM denotesthenumberofwindows.
| ComputetheglobalautocorrelationR |     |     |     | SS  | (l)forallpossiblelagsl∈[0:M] |     |     |     |     |     |
| -------------------------------- | --- | --- | --- | --- | ---------------------------- | --- | --- | --- | --- | --- |
forallnumberoftrainingiterationsdo
)}N
| Sampleamini-batch{((X |     |     | n   | ,Y n ),T | n   | fromD |     |     |     |     |
| --------------------- | --- | --- | --- | -------- | --- | ----- | --- | --- | --- | --- |
n=1
Forward{X ,T }N andgetcorrespondingrepresentation{v}N andpredictions{Yˆ }N
|                                  |     | n n | n=1     |                                   |              |     |      | n=1              |      | n n=1 |
| -------------------------------- | --- | --- | ------- | --------------------------------- | ------------ | --- | ---- | ---------------- | ---- | ----- |
| Computewindowrelationshipmatrixr |     |     |         |                                   | ∈RN×N,r(i,j) |     |      | =R (|T(0)−T(0)|) |      |       |
|                                  |     |     |         |                                   |              |     |      | SS i             | j    |       |
| ComputeAutoConlossL              |     |     |         | followingEquation3asinputs{v(n)}N |              |     |      |                  | andr |       |
|                                  |     |     | AutoCon |                                   |              |     |      |                  | n=1  |       |
|                                  |     |     |         |                                   | 1 (cid:80)N  | (Yˆ | )2   |                  |      |       |
| ComputeforecastinglossL          |     |     |         | mse =                             |              | n   | −Y n |                  |      |       |
N n=1
| DoonetrainingstepusingthefulllossL=L |     |     |     |     |     |     | +λ·L |         |     |     |
| ------------------------------------ | --- | --- | --- | --- | --- | --- | ---- | ------- | --- | --- |
|                                      |     |     |     |     |     | mse |      | AutoCon |     |     |
endfor
OurredesignedmodelandAutoConwereimplementedbasedontheTSlibcoderepository1. Our
sourcecodecanbeaccessedatazipfileinthesupplementary.
A.2 DETAILSOFMULTIVARIATEFORECASTING
Therearetworepresentativeapproachestomultivariateforecasting: Channel-mixingandChannel-
independenceapproaches. Thechannel-mixingapproachinvolvesmappingthevaluesofmultiple
channels at the same step into an embedding space and extracting temporal dependencies from
thisembeddingsequence. Thisapproachhasbeenadoptedbyvariouspapers(Zhouetal.,2022b;
Wu et al., 2023; Zhang & Yan, 2023). The channel-independence approach, on the other hand,
preserveseachchannel’sinformationwithoutmixingthemandlearnstemporalpatternswithineach
channelindependently. Recently, thismethodhasbeenadoptedinhigh-performingmodelssuch
asPatchTST(Challuetal.,2023)andLinearmodels(Zengetal.,2023), demonstratingsuperior
performanceoncurrentbenchmarkdatasets. Intermsofimplementation,eachchannelistreated
as a batch axis for computations. This effectively increases the amount of training data by the
numberofchannels,andthemodelparametersaresharedacrossmultiplechannels. Followingthe
channel-independenceapproach,wefirstcomputeautocorrelationsforeachchannelseparatelyin
ordertocalculateAutoCon. Wethentraintherepresentationtailoredtoeachchannelbasedonthese
autocorrelations.
A.3 DETAILSOFDATASETS
Inthispaper,weutilizedsixreal-worlddatasetsfromdiversedomains: mechanicalsystems(ETT),
energy(Electricity),traffic(Traffic),weather(Weather),economics(Exchange),anddisease(ILI).
ThestatisticsofeachdatasetaresummarizedandfoundinTable5. Asamainstreambenchmark,the
ETTdatasetsareextensivelyutilizedforassessinglong-termforecastingmethodsZhouetal.(2021);
Wuetal.(2021);Zhouetal.(2022b);Zengetal.(2023);Wuetal.(2023). ETTconsistsofcritical
1https://github.com/thuml/Time-Series-Library
13

PublishedasaconferencepaperatICLR2024
Table5: Statisticsofninedatasets.
Dataset Domain Timeserieslength Thenumberofvariables Samplingfrequency
| ETTh1 SystemMonitoring        | 17420 | 7   | 1Hour     |
| ----------------------------- | ----- | --- | --------- |
| ETTh2 SystemMonitoring        | 17420 | 7   | 1Hour     |
| ETTm1 SystemMonitoring        | 69680 | 7   | 15minutes |
| ETTm2 SystemMonitoring        | 69680 | 7   | 15minutes |
| Electricity EnergyConsumption | 26304 | 321 | 1Hour     |
| Traffic Traffic               | 17544 | 862 | 1Hour     |
| Weather Weather               | 52695 | 21  | 10Minutes |
| Exchange Economic             | 7588  | 8   | 1Day      |
| ILI Medical                   | 966   | 7   | 1Week     |
indicators(suchasoiltemperature,load,andothers)thataregatheredoveraspanoftwoyearsfrom
electricitytransformers. Thesedatasetsaregroupedintofourdistinctsetsbasedonlocation(ETT1
andETT2)andtimeinterval(15minutesandonehour). TheElectricitydatasetcapturesthehourly
electricityconsumptionof321customersfrom2012to2014. Ontheotherhand,theTrafficdataset
compileshourlydatafromtheCaliforniaDepartmentofTransportation,detailingtheoccupancyrates
ofroadsasmeasuredbydifferentsensorsonfreewaysintheSanFranciscoBayarea. TheWeather
dataset consists of 21 meteorological indicators, including air temperature, humidity, and others,
recordedat10-minuteintervalsoverthecourseofayear. TheExchangedatasetchroniclesdaily
exchangeratesofeightdifferentcountriesfrom1990through2016. Lastly,theILIdatasetincludes
weeklyrecordsofinfluenza-likeillness(ILI)patientdatafromtheCentersforDiseaseControland
PreventionintheUnitedStates,spanningfrom2002to2021. Thisdatasetillustratestheratioof
patientsdiagnosedwithILIrelativetothetotalpatientcount.
A.4 BASELINEMODELS
In the realm of long-term forecasting, numerous models have been proposed since the advent
ofInformer. Thesemodelshavedemonstratedcommendableperformancewithuniquenovelties.
However,theywerecomparedwithunderperformingmodelssuchasthemodelsbasedonRNNs,
andTransformer-basedmodels, whichareknowntobesusceptibletooverfitting. Therefore, our
primaryfocusisonhigh-performingandstate-of-the-artmodelsamongthemostrecentproposals.
Wevalidateourmethodagainstsevenforecastingbaselinesandthreerepresentationmethodologies.
AllmodelswereimplementedusingPyTorch. Forthelatestforecastingmodels,namelyTimesNet2,
DLinearandNLinear3,MICN4,FiLM5,NonstationaryTransformer6,andFEDformer7,weutilized
theofficialcodereleasedbytheoriginalauthorsratherthanimplementingitfromscratch.
Similarly,forrecentrepresentationmethods,suchasLaST8,CoST9,andTS2Vec10,weutilizedthe
officialcodesprovidedbytheauthorsinsteadofimplementingmodelsfromscratch. Weadheredto
theuniquehyperparametersofeachmodel,tuningwithintheparametersearchrangethatyielded
optimalperformance. However,certainconfigurations,suchasinputlengthandoutputlength,were
setuniformlyforeaseofcomparison. Morespecificevaluationprotocolswillbepresentedinthe
followingsection.
2https://github.com/thuml/Time-Series-Library
3https://github.com/cure-lab/LTSF-Linear
4https://github.com/wanghq21/MICN
5https://github.com/DAMO-DI-ML/NeurIPS2022-FiLM
| 6https://github.com/thuml/Nonstationary | Transformers |     |     |
| --------------------------------------- | ------------ | --- | --- |
7https://github.com/MAZiqing/FEDformer
8https://github.com/zhycs/LaST
9https://github.com/salesforce/CoST
10https://github.com/yuezhihan/ts2vec
14

PublishedasaconferencepaperatICLR2024
A.5 EVALUATIONDETAILS
Inourexperiments,weaimtoassessthemodel’scapabilitytocapturelong-termvariations,sothat
theoutputlengthshouldbelongenoughtobeaffectedbythesevariations. Increasingtheoutput
lengthmorethanthoseusedinpreviousexperiments,however,involvesseveralconsiderations. Thus,
wedelineatethemodificationstothestandardevaluationprotocolasfollows:
1. TheinputlengthI issetto14(fortheILIdataset),48(fortheExchangedataset),192(for
theETTmdataset),and96(fortheothersdatasets). Theseinputlengthsallowustoincrease
theoutputlengthwithinthelimitedwindowlength,inaccordancewiththetotallengthof
eachdataset.
2. Thestandardprotocolsplitsalldatasetsintotraining,validation,andtestsetsinchronological
order,witharatioof6:2:2fortheETTdatasetand7:1:2fortheremainingdatasets.However,
duetotheincreasedwindowlengthintheotherdatasets,thevalidationsetisinsufficiently
| populated. | Therefore,weadoptaratioof6:2:2foralldatasets. |     |     |     |     |
| ---------- | --------------------------------------------- | --- | --- | --- | --- |
3. Theweatherdatasetcontainsnegativevaluesforindicatorsthatshouldlogicallybenon-
negative. Theseerroneouslabels,ifnotcorrected,couldimpedeaccurateevaluationdueto
| scalingissues. | Werectifiedtheseerrorsbyfillingthemwithneighboringvalues. |     |     |     |     |
| -------------- | --------------------------------------------------------- | --- | --- | --- | --- |
Weadheredtostandardprotocolsforallexperiments,barringtheexceptionsoutlinedabove.
A.6 HYPERPARAMETERSENSITIVITY
|     | MSE |     |     | MAE |     |
| --- | --- | --- | --- | --- | --- |
0.04
0.05
| etaR noitcudeR |     | etaR noitcudeR | 0.02       |          |     |
| -------------- | --- | -------------- | ---------- | -------- | --- |
| 0.00           |     |                | 0.00       |          |     |
| 0.05           |     |                | 0.02       |          |     |
|                |     |                | 0.04 ETTh1 | Weather  |     |
| 0.10           |     |                | ETTh2      | Exchange |     |
|                |     |                | ECL        | illness  |     |
0.06 Traffic
| 0.0 1e-4 | 1e-3 1e-2 | 1e-1 1.0 | 0.0 1e-4 | 1e-3 1e-2 | 1e-1 1.0 |
| -------- | --------- | -------- | -------- | --------- | -------- |
|          | Lambda    |          |          | Lambda    |          |
Figure9: ErrorreductionratesofsevendatasetsaccordingtothelambdavalueofAutoCon: (Left)
Meansquarederrorand(Right)Meanabsoluteerror. Theerrorreductionrateswerecalculatedbased
onthelambda,whichissetat0.0asthereferencepoint. Thebestperformancesinalldatasetswere
achievedwhenusingourAutoCon(denotedthestarmarkers). Itisimportanttonotethatahigh
lambdavaluedoesnotnecessarilyimplystronglong-termvariations.
| B ADDITIONAL | EXPERIMENTS |     |     |     |     |
| ------------ | ----------- | --- | --- | --- | --- |
B.1 EXPERIMENTSONWINDOWLENGTH
AsmentionedinSection5,weconsideredsimplyincreasingthelengthofthewindowtocapturethe
long-termvariationsasmuchaspossible. Also,asthewindowlengthincreases,thenumberofdata
windowsusedforlearningdecreases. Afterall,wehypothesizethatincreasingthewindowlength
increasestheinputcomplexityofthemodel,whilereducingthenumberofdatapoints,makingthe
modelvulnerabletooverfitting.
Figures10and11depictthetrainingandtestinglosseswhentheinputlengthincreasefrom192to
920forthepurposeofpredicting720stepsinETTh1andETTh2,respectively. Weobservedthat
theoveralltestlosstendstosoarorconverge,whilethetraininglosspersistentlydecreaseswhenthe
inputsizeisincreasedinfivemodelswithvaryingcapacitiesandattributes.
15

PublishedasaconferencepaperatICLR2024
|      | DLinear |      | TimesNet |      | MICN | Nonstationary_Transformer |       | FEDformer |     |
| ---- | ------- | ---- | -------- | ---- | ---- | ------------------------- | ----- | --------- | --- |
| 0.22 |         | 0.12 |          |      |      |                           |       |           |     |
|      |         |      |          | 0.35 |      | 0.16                      | 0.275 |           |     |
0.20
|      |     | 0.10 |     | 0.30 |     | 0.15 | 0.250 |     |     |
| ---- | --- | ---- | --- | ---- | --- | ---- | ----- | --- | --- |
| 0.18 |     |      |     |      |     | 0.14 | 0.225 |     |     |
0.25
| ESM 0.16 | train | 0.08 |     |      |     | 0.13    | 0.200     |     |     |
| -------- | ----- | ---- | --- | ---- | --- | ------- | --------- | --- | --- |
|          | test  |      |     |      |     | 0.12    |           |     |     |
| 0.14     |       |      |     | 0.20 |     |         | 0.175     |     |     |
|          | ours  | 0.06 |     |      |     | 0.11    | 0.150     |     |     |
| 0.12     |       |      |     | 0.15 |     |         |           |     |     |
|          |       | 0.04 |     |      |     | 0 . 1 0 | 0 . 1 2 5 |     |     |
| 0.10     |       |      |     | 0.10 |     | 0 . 0 9 |           |     |     |
0 . 1 0 0
| 0.08 |     | 0.02    |     | 0.05    |     | 0.08        | 0.075 |         |     |
| ---- | --- | ------- | --- | ------- | --- | ----------- | ----- | ------- | --- |
| 192  | 480 | 960 192 | 480 | 960 192 | 480 | 960 192 480 | 960   | 192 480 | 960 |
Input Length
Figure10: Thefigureshowsthetraining(blueline)andtest(orangeline)losstraceplotsoffive
models as the input length gradually increases on the ETTh1 with the output-720 setting. In the
figure,theredhorizontallinerepresentstheperformanceofourmodelwiththeinput-96setting.
|      | DLinear |       | TimesNet |      | MICN | Nonstationary_Transformer |      | FEDformer |     |
| ---- | ------- | ----- | -------- | ---- | ---- | ------------------------- | ---- | --------- | --- |
|      |         | 0.200 |          |      |      | 0.24                      |      |           |     |
| 0.30 |         |       |          | 0.35 |      |                           | 0.40 |           |     |
|      |         | 0.175 |          |      |      | 0.23                      |      |           |     |
| 0.28 |         |       |          | 0.30 |      |                           | 0.35 |           |     |
|      |         | 0.150 |          |      |      | 0.22                      |      |           |     |
0.26
| ESM  |       | 0.125   |     | 0.25    |     | 0.21        | 0.30 |         |     |
| ---- | ----- | ------- | --- | ------- | --- | ----------- | ---- | ------- | --- |
| 0.24 |       |         |     |         |     | 0.20        |      |         |     |
| 0.22 |       | 0.100   |     | 0.20    |     |             |      |         |     |
|      |       |         |     |         |     | 0.19        | 0.25 |         |     |
| 0.20 | train | 0.075   |     | 0.15    |     |             |      |         |     |
|      | test  |         |     |         |     | 0.18        |      |         |     |
| 0.18 |       | 0.050   |     |         |     |             | 0.20 |         |     |
|      | ours  |         |     | 0.10    |     | 0.17        |      |         |     |
| 0.16 |       | 0.025   |     |         |     |             |      |         |     |
| 192  | 480   | 960 192 | 480 | 960 192 | 480 | 960 192 480 | 960  | 192 480 | 960 |
Input Length
Figure11: Thefigureshowsthetraining(blueline)andtest(orangeline)losstraceplotsoffive
models as the input length gradually increases on the ETTh2 with the output-720 setting. In the
figure,theredhorizontallinerepresentstheperformanceofourmodelwiththeinput-96setting.
Also,inthecaseofDLinearinFigure11,bothtestandtraininglossesdeclineinsyncduetolimited
capacity.However,weregarditasanunderfittingproblemsincetesterrorsarehigherthanourmethod
(see red line). Consequently, we empirically substantiate that merely increasing the input length
doesnotnecessarilyenhancetheperformanceoflong-termforecasting. Moreover,itisnoteworthy
thatthecomputationalcostforcomplexmodelsotherthanlinearmodelsincreasessignificantlywith
increasingsequencelength.
B.2 ADDITIONALFIGURE6RESULTSONOTHERDATASETS
Additionally,weprovideresultsforETTh1(Figure12)andElectricity(Figure13),whichshowthe
long-termvariations. Althoughtherearesomedifferencesinmagnitude,theoveralltrendsaresimilar
acrossthethreedatasets.
|     |     | (a) Comparison with baselines |     |     |     | (b) Ablations in our method |     |     |     |
| --- | --- | ----------------------------- | --- | --- | --- | --------------------------- | --- | --- | --- |
|     | E   |                               |     |     |     | E                           |     |     |     |
|     | S M |                               |     |     |     | S M                         |     |     |     |
I/O ratio = 1
I/O ratio = 1
|     |     |     | Prediction Length O |     |     |     | Prediction Length O |     |     |
| --- | --- | --- | ------------------- | --- | --- | --- | ------------------- | --- | --- |
Figure12: Comparisonofforecastingerror(MSE)accordingtopredictionlengthO. (a)comparison
withbaselinemodelsand(b)comparisonbetweenablationsofourmethodontheETTh1dataset.
16

PublishedasaconferencepaperatICLR2024
(a) Comparison with baselines (b) Ablations in our method
E E
S S
M M
I/O ratio = 1 I/O ratio = 1
Prediction Length O Prediction Length O
Figure13: Comparisonofforecastingerror(MSE)accordingtopredictionlengthO. (a)comparison
withbaselinemodelsand(b)comparisonbetweenablationsofourmethodontheElectricitydataset.
Table6: Increasingcomplexityoflong-termbranchinDLinear.
ETTh1 ETTh2 Electricity
Model #oflayers
MSE MAE MSE MAE MSE MAE
DLinear 1 0.1780±0.0054 0.3466±0.0063 0.2929±0.0140 0.4362±0.0087 0.3067±0.01544 0.4125±0.0109
2 0.1993±0.1964 0.3638±0.2191 0.3170±0.0488 0.4581±0.0404 0.3775±0.01574 0.4618±0.0081
3 0.2793±0.0678 0.4594±0.0682 0.3008±0.0046 0.4451±0.0035 0.3057±0.00902 0.4097±0.0065
4 0.2760±0.0663 0.4564±0.0683 0.3015±0.0031 0.4455±0.0024 0.3165±0.03838 0.4183±0.0279
B.3 ADDITIONALABLATIONRESULTSFORLONG-TERMBRANCH
Increasingthecomplexityofthelong-termbranchisessentialforlearninglong-termrepresentations,
butitisnotthesolereasonforthesuperiorityofourmethodology. Inotherwords,evenwiththe
increasedcomplexity,capturinglong-termvariationisnoteasyinthecurrentframeworkthatuses
onlyforecastingloss. Asthemaincontribution,usingAutoConisessentialforlearninglong-term
variation and leading to performance improvement. To verify this, we additionally present two
ablationresults: increasingcomplexityinDLinearandinourmodel.
First, DLinearutilizesonlyasinglelinearlayerforbothlong-termandshort-termbranches. We
increasethecomplexityofthelong-termbranchbystackinglinearlayerswithanactivationfunction
inthelong-termbranch. However,asevidentinTable6below,evenwhenstackinglayersinthelong
term,performancetendstodecreaseorremainsimilar. Thisdemonstratesthatincreasinglong-term
complexityisnoteffectiveintheexistingdecompositionarchitecture.
Second, the following Table 7 demonstrates the performance changes based on the complexity
ofthelong-termbranchinourdecompositionarchitecture. WithoutAutocon,ourmodelmaybe
slightlybetterorcomparabletothesecond-bestmodel. Thehighestperformanceisachievedonly
whenAutoConisemployed. ThisfurtherunderscoresthenecessityoftheAutoConweproposedto
accuratelypredictlong-termvariation.
B.4 ANALYSISOFCOMPUTATIONALCOST
Giventhereal-timenatureofmosttime-seriesapplications,computationalefficiencyisacrucialfactor
intime-seriesforecasting(Dannecker,2015;Iqbaletal.,2019;Torresetal.,2021). Asforecasting
horizonsincrease,thewindowlengthexpands,leadingtoincreasedcomputationalcosts. Therefore,
itisimperativetoevaluateamodel’scomputationalefficiency. Figure14illustratesthetimerequired
toupdatetheparametersbyasinglebatchforourmodelincomparisonwiththebaselinemodels.
Thecomputationalcostwasmeasuredforfourdifferentoutputlengths,rangingfrom96to2160,for
eachmodel. Abatchsizeof32wasused,andallmeasurementsweretakenindependentlyinthe
sameGPUandserverenvironment. Firstly,thelinearmodeltooktheleastamountoftimeduetoits
minimalnumberofparametersandthesimplicityofmatrixmultiplicationoperations. Ontheother
hand,TimesNetrequiredthemosttimeasitextractsmultipleperiodsandcomputesaloopforeach
period.TheNonstationarymodel,whichisbasedontheTransformer,hasacomputationalcomplexity
ofO(W2)inrelationtolength,whichexplainsthesharpincreaseincomputationtimewithlength.
Overall,ourmodelwasthesecondfastestafterthelinearmodel,anditscomputationcostdidnot
17

PublishedasaconferencepaperatICLR2024
| Table7: | Increasingcomplexityoflong-termbranchinourmodel. |     |     |     |     |
| ------- | ------------------------------------------------ | --- | --- | --- | --- |
ETTh1
| Model                  | #oflayers | AutoCon |               |               |     |
| ---------------------- | --------- | ------- | ------------- | ------------- | --- |
|                        |           |         | MSE           | MAE           |     |
| TimesNet(baselinebest) |           | 2 -     | 0.0834±0.0024 | 0.2310±0.0023 |     |
| Ours                   |           | 1 X     | 0.0837±0.0185 | 0.2372±0.0294 |     |
| Ours                   |           | 2 X     | 0.0910±0.0188 | 0.2360±0.0257 |     |
| Ours                   |           | 3 X     | 0.0876±0.0240 | 0.2351±0.0303 |     |
| Ours                   |           | 4 X     | 0.0918±0.0194 | 0.2381±0.0241 |     |
| Ours(best)             |           | 1 O     | 0.0787±0.002  | 0.2226±0.0023 |     |
ETTh2
| Model                  | #oflayers | AutoCon |               |               |     |
| ---------------------- | --------- | ------- | ------------- | ------------- | --- |
|                        |           |         | MSE           | MAE           |     |
| TimesNet(baselinebest) |           | 2 -     | 0.2074±0.0113 | 0.3703±0.0155 |     |
| Ours                   |           | 1 X     | 0.2023±0.0230 | 0.3594±0.0261 |     |
| Ours                   |           | 2 X     | 0.2020±0.0098 | 0.3525±0.0078 |     |
| Ours                   |           | 3 X     | 0.2036±0.0265 | 0.3573±0.0224 |     |
| Ours                   |           | 4 X     | 0.2087±0.0167 | 0.3605±0.0130 |     |
| Ours(best)             |           | 3 O     | 0.1771±0.0393 | 0.3441±0.0366 |     |
Electricity
| Model                 | #oflayers | AutoCon |               |               |     |
| --------------------- | --------- | ------- | ------------- | ------------- | --- |
|                       |           |         | MSE           | MAE           |     |
| DLinear(baselinebest) |           | 1 -     | 0.3067±0.0154 | 0.4125±0.0109 |     |
| Ours                  |           | 1 X     | 0.2928±0.1369 | 0.3978±0.1009 |     |
| Ours                  |           | 2 X     | 0.2889±0.0330 | 0.3927±0.0209 |     |
| Ours                  |           | 3 X     | 0.2975±0.0458 | 0.4049±0.0481 |     |
| Ours                  |           | 4 X     | 0.3089±0.4737 | 0.4115±0.0587 |     |
| Ours(best)            |           | 2 O     | 0.2753±0.0224 | 0.3861±0.0166 |     |
466.1 ms/iter
96 > 96
96 > 720
96 > 1440
400
96 > 2160
365.7 ms/iter
)reti/sm( tsoC lanoitatupmoC
300
269.4 ms/iter
200
| 100 |     | 91.2 ms/iter |     |     |     |
| --- | --- | ------------ | --- | --- | --- |
35.8 ms/iter
33.2 ms/iter 31.1 ms/iter
|     | 4.5 ms/iter | 5.1 ms/iter |     |     |     |
| --- | ----------- | ----------- | --- | --- | --- |
0
| oCon) toCon) | NLinear DLinear | FiLM | TimesNet | MICN Nonstationary | FEDformer |
| ------------ | --------------- | ---- | -------- | ------------------ | --------- |
O u r s u rs
 A u t O  A u
(w/ (w/o
Figure14: Thefigureillustratesthecomparisonofcomputationalcosts,measuredinmillisecondsper
asinglebatchiteration(ms/iter),ofthebaselinemodelsandourmodel. Thecomputationalcostfor
eachmodelwasevaluatedbyincreasingtheoutputlengthfrom96to2160.
significantly increase even when training included AutoCon (from 31.1 ms/iter to 33.2 ms/iter).
Hence,ourmethodmanagestoachievesuperiorlong-termforecastingperformancecomparedto
18

PublishedasaconferencepaperatICLR2024
linearmodels,whilerequiringlesscomputationalresourcesthanothermorecomplexmodels. The
costanalysiswasbrieflyaddressedinSection4.4ofthemainpaper.
B.5 VISUALIZATIONOFFORECASTINGRESULTS
0.5
0.0
0.5
1.0
1.5
NLinear (2023), MSE:0.4291, Trend MSE:0.3493
2.0
0.5
0.0
0.5
1.0
1.5
DLinear (2023), MSE:0.3924, Trend MSE:0.3168
2.0
0.5 TimesNet (2023), MSE:0.1697, Trend MSE:0.0878
0.0
0.5
1.0
1.5
2.0
0 200 400 600 800 1000 1200 1400
0.5 Nonstaionary (2022), MSE:0.1852, Trend MSE:0.0575
0.0
0.5
1.0
1.5
2.0
0 200 400 600 800 1000 1200 1400
0.5 Ours, MSE:0.1263, Trend MSE:0.0493
0.0
0.5
1.0
1.5
2.0
0 200 400 600 800 1000 1200 1400
Figure 15: The figure displays the visualization of forecasting results for the 96-1440 setting in
ETTh2,showcasingourmodelalongwithfourothermodels.
Figure 15 provides a qualitative result of the five different models by visualizing the prediction
results for 1440 steps in the ETTh2 dataset. In the case of linear models, the error increases as
the prediction distance increases, failing to account for long-term variations. Nonstationary and
TimesNetmodels,althoughbetteratfollowinglong-termvariationsthanthelinearmodel,struggleto
capturehigh-frequencypatternseffectively. Ourmodel,ontheotherhand,successfullymanagesto
capturebothlong-termvariationsandhigh-frequencypatterns. Thiscanbeattributedtoourmodel’s
structure,whichisdesignedtobenefitfrombothshort-termandlong-termpredictions.
B.6 EVALUATIONRESULTSWITHOTHERMETRICS
Whileexistingmetrics(i.e.,MSEandMAE)arestandardmetricsforlong-termforecastingevaluation,
they have limitations. Specifically, they may not adequately capture aspects such as the shape
andtemporalalignmentofthetimeseries,whicharecrucialforacomprehensiveevaluationofa
forecastingmodel’sperformance.
Toaddresstheselimitations,weintroducedtwoadditionalmetricsbasedonDynamicTimeWarping
(DTW) (Sakoe & Chiba, 1978): Shape DTW and Temporal DTW (Le Guen & Thome, 2019).
ShapeDTWfocusesonthesimilarityofthepatternorshapeofthepredictedsequencetotheactual
sequence, providinginsightintothemodel’sabilitytocapturetheunderlyingpatternofthetime
series. TemporalDTWevaluatesthealignmentofthepredictedsequencewiththeactualsequence,
highlightingthemodel’saccuracyinforecastingthetimingofevents.
Theseadditionalmetricsofferamorenuancedassessmentofourmodel’sperformance,particularly
inareaswhereMSEandMAEmayfallshort. LowervaluesinbothShapeDTWandTemporalDTW
indicatebetterperformance,signifyinglesserdistortionbetweenthepredictedandactualsequences.
AsshowninTable8,ourmethoddemonstratessuperiorperformancenotonlyinMSEandMAEbut
alsointheseshapeandtemporalalignment-focusedmetrics.
19

PublishedasaconferencepaperatICLR2024
Table8: Univariateforecastingresults,evaluatedusingShapeandTemporalDTWina720-output
settingacrossthreedatasets
Dataset ETTh1 ETTh2 Electricity
Model\Metric ShapeDTW TemporalDTW ShapeDTW TemporalDTW ShapeDTW TemporalDTW
Ours 17.14±1.653 59.66±1.739 42.38±0.630 13.47±1.793 80.73±7.798 0.09±0.014
TimesNet 25.80±4.349 86.86±15.509 62.58±5.390 51.19±21.834 139.83±16.516 0.49±0.826
PatchTST 22.21±1.226 72.23±4.411 65.45±2.976 15.23±0.882 116.50±29.878 0.75±1.674
MICN 37.08±12.393 65.70±3.588 67.69±8.796 22.67±3.168 123.93±17.543 1.90±1.176
DLinear 58.32±1.955 155.21±8.587 82.53±3.627 24.88±2.403 88.70±3.550 0.18±0.145
C ANALYSIS OF REPRESENTATION
C.1 DETAILSOFREPRESENTATIONSIMILARITIES
InFigure2,weusedthreebaselineswithourmodelandextractedtherepresentationsofeachbaseline
eitherbeforethefinalprojectionlayer(TimesNet)oraftertheencoderlayer(PatchTST,FEDformer,
andOurs). Ourmainpurposeinvisualizingtherepresentationwastodemonstratethelearningof
long-termcorrelations. Todisplaymoreclearly,weappliedafilteringmethodtosmoothshort-term
fluctuationswithinthegivenwindow. Wealsoprovideoriginalrepresentationresults, whichare
enlargedforeachbaseline,withoutsmoothingouttheshort-termfluctuationsasshowninFigure16.
Figure16showsthatthethreebaselinemodelslearntheshort-termcorrelationswithinthewindow,
althoughtheydonotlearnthelong-termcorrelations.
Oneinterestingpointinthisfindingisthatexistingmodelshaveattemptedtoaddressthelimitations
ofwindowlengthbyleveragingtime-stampinformation. Actually,TimesNet,FEDformer,andour
modelobtainedtherepresentationsusingtimestamps,incorporatingthemintotheinputsequences,
whilePatchTSTdoesnotutilizethetimestamps. However,notonlyPatchTSTbutalsobothTimesNet
andFEDformerdonoteffectivelycapturetheannualcyclicpatterns,despiteutilizingtimestamps
asthe sameas ourmodel. These failures arenoteworthy, particularly consideringthe Electricity
timeseries,whichdisplaysayearly-longperiodicity. Theseresultsshowthatitischallengingfor
themodeltolearnyearlypatternsevenwhengiveninputsequencesandtimestamps,solelyrelying
ontheexistingforecastingloss. Therefore,thisresultdemonstratesthenecessityofourAutoCon
loss. Furthermore,tojustifytheemergenceoflong-termrepresentationirrespectiveofthemodel’s
structuralaspects,weprovideadditionalresultsoftherepresentationfromanablationmodelthat
doesnotutilizeAutoConinourmodel. AsshowninFigure17,themodelwithoutAutoConalso
exhibitsaweakperiodicity,butsimilartootherbaselines,representationsimilarityremainsrelatively
flatcomparedtoourfullmodel.
C.2 VISUALIZATIONOFREPRESENTATIONS
WebenchmarkedourAutoConmethodagainstthreerepresentationlearningmethodsproposedto
enhanceforecastingperformance. TS2VecandCoSThaveatwo-stagelearningframeworkinwhich
they utilize a ridge regression model for time-series forecasting and deep learning-based models
forrepresentationlearning. Ontheotherhand,LaSTandourmethodadoptanend-to-endlearning
frameworkwhereinbothrepresentationandtime-seriesforecastinglearningoccurconcurrently.
Figure7presentstherepresentationresultsoffourmethodsovertheETTh2dataset. Toinvestigate
whethereachmethodhaslearnedtherepresentationstructureassociatedwiththelong-termvariations,
weextractedrepresentationscorrespondingtoalltrainingtimestepsandvisualizedthemviaUMAP
withthemonthlabelsderivedfromthetimestamp. Ourmodelclearlydisplayscontinuitybetween
adjacentmonthsanddemonstrateswell-definedclustering,attributesnotseenintheothermodels. It
seemsthattheothermodelsdidnotlearnthestructurenecessaryforrecognizingone-yearlong-term
variationsbeyondthewindow,astheirrepresentationlearningwasconfinedwithinthewindow.
20

PublishedasaconferencepaperatICLR2024
| 𝑤     | Yearly-long periodicity  |      | 𝑤        | 𝑤   |
| ----- | ------------------------ | ---- | -------- | --- |
|       | 1                        |      | 2        | 3   |
| 𝑰nput | 𝑶utput                   |      |          |     |
|       | 𝑆𝑖𝑚(𝑤                    | ,𝑤 ) | 𝑆𝑖𝑚(𝑤 ,𝑤 | )   |
|       |                          | 2 1  | 2        | 3   |
Figure16:(Top)Electricitytimeseriesincludingalong-termvariationbeyondwindowsize.(Bottom)
PlottedrepresentationsimilaritiesofthreebaselinemodelsbetweenananchorwindowW andall
2
| otherwindowsincludingW | andW                       | .    |       |      |
| ---------------------- | -------------------------- | ---- | ----- | ---- |
|                        | 1                          | 3    |       |      |
| 𝑤                      |                            |      | 𝑤     | 𝑤    |
|                        | 1 Yearly-long periodicity  |      | 2     | 3    |
| 𝑰nput                  | 𝑶utput                     |      |       |      |
|                        | 𝑆𝑖𝑚(𝑤                      | ,𝑤 ) | 𝑆𝑖𝑚(𝑤 | ,𝑤 ) |
|                        | 2                          | 1    | 2     | 3    |
Figure17:(Top)Electricitytimeseriesincludingalong-termvariationbeyondwindowsize.(Bottom)
Plottedrepresentationsimilaritiesofourmodels(withAutoConandwithoutAutoCon)betweenan
| anchorwindowW | andallotherwindowsincludingW |     | andW | .   |
| ------------- | ---------------------------- | --- | ---- | --- |
|               | 2                            |     | 1    | 3   |
C.3 ADDITIONALCOMPARISONWITHTWOSELF-SUPERVISEDLOSS
Wedesignedandprovidedresultsfortwopossibleself-supervisedobjectivesbasedonHierCon(Yue
etal.,2022)andSupCon(Khoslaetal.,2020)thatcanbeincorporatedintoourtwo-streammodel
structure. HierConinducestherepresentationsoftwopartiallyoverlappedwindowstobecloseto
eachother,whileSupConencouragestheencodertolearncloserepresentationsforthewindowswith
thesamemonthlabel.
ThetwoSSLobjectivesweretestedwithourmodelarchitecture,replacingonlytheAutoConloss. As
showninTable9,comparedtotheonewithoutanySSLloss,HierConshowsaslightimprovement
in performance on the ETTh1 and ETTh2 for short-term predictions with a length of 96, but it
performs worse in the long-term prediction, as it emphasizes only temporal closeness. SupCon
leveragedmonthlyinformationbeyondthewindowlength,leadingtoperformanceimprovements
eveninthelong-termprediction. However,SupConcanonlylearnasinglepredefinedperiodicity,
unlikeAutoCon. Consequently,SupConshowslowerperformancethanAutoConinlearningthe
periodicitiesexistinginthetimeseriesthroughautocorrelation.
D FULL
BENCHMARKS
Table10andTable11showthefullbenchmarkresultsincludingconfidenceintervals.
21

PublishedasaconferencepaperatICLR2024
Table9: Comparisonwithdifferentself-supervisedobjectivesinourredesignedarchitecture.
| Datasets |     | ETTh1 |     | ETTh2 |     | Electricity |     |
| -------- | --- | ----- | --- | ----- | --- | ----------- | --- |
Predictionlength 96 720 1440 2160 96 720 1440 2160 96 720 1440 2160
|     | MSE 0.061 | 0.082 0.096 | 0.130 0.147 | 0.214 0.213 | 0.236 | 0.206 0.289 | 0.363 0.419 |
| --- | --------- | ----------- | ----------- | ----------- | ----- | ----------- | ----------- |
Oursw/oSSL
|     | MAE 0.190 | 0.226 0.245 | 0.286 0.285 | 0.375 0.365 | 0.372 | 0.322 0.393 | 0.460 0.503 |
| --- | --------- | ----------- | ----------- | ----------- | ----- | ----------- | ----------- |
|     | MSE 0.059 | 0.090 0.121 | 0.145 0.132 | 0.221 0.221 | 0.263 | 0.223 0.313 | 0.380 0.500 |
Oursw/HierCon
|     | MAE 0.187 | 0.240 0.276 | 0.306 0.279 | 0.379 0.378 | 0.407 | 0.339 0.407 | 0.474 0.544 |
| --- | --------- | ----------- | ----------- | ----------- | ----- | ----------- | ----------- |
|     | MSE 0.056 | 0.082 0.092 | 0.091 0.125 | 0.185 0.199 | 0.215 | 0.209 0.279 | 0.351 0.408 |
Oursw/SupCon
|     | MAE 0.184 | 0.229 0.242 | 0.237 0.270 | 0.349 0.360 | 0.371 | 0.326 0.388 | 0.451 0.489 |
| --- | --------- | ----------- | ----------- | ----------- | ----- | ----------- | ----------- |
|     | MSE 0.056 | 0.079 0.079 | 0.074 0.124 | 0.177 0.176 | 0.198 | 0.196 0.275 | 0.338 0.380 |
Oursw/AutoCon
|     | MAE 0.182 | 0.223 0.225 | 0.215 0.269 | 0.344 0.340 | 0.358 | 0.313 0.386 | 0.441 0.481 |
| --- | --------- | ----------- | ----------- | ----------- | ----- | ----------- | ----------- |
Table10: MultivariateforecastingresultsonETTdatasetswithdifferentpredictionlengthsO ∈
{96,192,336,720}withtheinputlengthI =96. ThefullbenchmarkisavailableatAppendixD
|     | TimesNet* | MICN | Crossformer | N-HiTS | DLinear* | ETSformer* | LightTS* |
| --- | --------- | ---- | ----------- | ------ | -------- | ---------- | -------- |
Models Ours
|     | (2023) | (2023) | (2023) | (2023) | (2023) | (2022) | (2022) |
| --- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
O MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE
96 0.387 0.396 0.384 0.402 0.421 0.431 0.407 0.429 0.404 0.424 0.386 0.400 0.494 0.479 0.424 0.432
1hTTE 192 0.437 0.424 0.436 0.429 0.474 0.487 0.505 0.496 0.465 0.466 0.437 0.432 0.538 0.504 0.475 0.462
336 0.476 0.442 0.491 0.469 0.569 0.551 0.620 0.574 0.519 0.501 0.481 0.459 0.574 0.521 0.518 0.488
720 0.468 0.461 0.521 0.500 0.770 0.672 0.830 0.701 0.621 0.569 0.519 0.516 0.562 0.535 0.547 0.533
Average 0.442 0.431 0.458 0.450 0.559 0.535 0.591 0.550 0.502 0.490 0.456 0.452 0.542 0.510 0.491 0.479
96 0.290 0.341 0.340 0.374 0.299 0.364 0.645 0.562 0.346 0.381 0.333 0.387 0.340 0.391 0.397 0.437
2hTTE
192 0.373 0.398 0.402 0.414 0.441 0.454 0.788 0.636 0.427 0.440 0.477 0.476 0.430 0.439 0.520 0.504
336 0.408 0.423 0.452 0.452 0.654 0.567 0.959 0.709 0.518 0.500 0.594 0.541 0.485 0.479 0.626 0.559
720 0.419 0.442 0.462 0.468 0.956 0.716 1.146 0.784 0.888 0.645 0.831 0.657 0.500 0.497 0.863 0.672
Average 0.372 0.401 0.414 0.427 0.588 0.525 0.885 0.673 0.545 0.491 0.559 0.515 0.439 0.452 0.602 0.543
96 0.330 0.365 0.338 0.375 0.316 0.362 0.364 0.399 0.355 0.389 0.345 0.372 0.375 0.398 0.374 0.400
1mTTE 192 0.371 0.384 0.374 0.387 0.363 0.390 0.431 0.441 0.404 0.414 0.380 0.389 0.408 0.410 0.400 0.407
336 0.399 0.408 0.410 0.411 0.408 0.426 0.517 0.488 0.452 0.456 0.413 0.413 0.435 0.428 0.438 0.438
720 0.460 0.444 0.478 0.450 0.481 0.476 0.698 0.627 0.500 0.485 0.474 0.453 0.499 0.462 0.527 0.502
Average 0.390 0.400 0.400 0.406 0.392 0.414 0.503 0.489 0.428 0.436 0.403 0.407 0.429 0.425 0.435 0.437
96 0.178 0.260 0.187 0.267 0.179 0.275 0.272 0.357 0.201 0.287 0.193 0.292 0.189 0.280 0.209 0.308
2mTTE
192 0.244 0.303 0.249 0.309 0.307 0.376 0.335 0.414 0.295 0.354 0.284 0.362 0.253 0.319 0.311 0.382
336 0.305 0.341 0.321 0.351 0.325 0.388 0.564 0.590 0.359 0.391 0.369 0.427 0.314 0.357 0.442 0.466
720 0.398 0.396 0.408 0.403 0.502 0.490 1.203 0.779 0.530 0.498 0.554 0.522 0.414 0.413 0.675 0.587
Average 0.281 0.325 0.291 0.333 0.328 0.382 0.593 0.535 0.346 0.383 0.350 0.401 0.293 0.342 0.409 0.436
∗denotestheresultsaretakenfromTimesNet.
22

PublishedasaconferencepaperatICLR2024
Table11: Extendedlong-termforecastingfullbenchmarkswithconfidenceinterval. ‘S’hortand
‘L’ongindicatethelengthoftheconventionalexperimentandthenewlyextendedexperimentsetting,
respectively. Nonstationaryhasanout-of-memory(OOM)problemattheoutput-4320settingon
ETTm1andETTm2datasets.
Ours TimesNet(2023) MICN(2023) PatchTST(2023) DLinear(2023) FiLM(2023) Nonstationary(2022) FEDformer(2022)
Models
I→O MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE
0.056 0.182 0.058 0.185 0.062 0.185 0.057 0.184 0.063 0.185 0.057 0.180 0.069 0.197 0.080 0.218
96
S ±0.0015 ±0.0020 ±0.0014 ±0.0028 ±0.0020 ±0.0028 ±0.0017 ±0.0038 ±0.0038 ±0.0052 ±0.0006 ±0.0010 ±0.0068 ±0.0099 ±0.0037 ±0.0039
1hTTE
0.079 0.223 0.083 0.230 0.175 0.342 0.089 0.236 0.180 0.348 0.097 0.245 0.117 0.272 0.130 0.285
720 ±0.0085 ±0.0072 ±0.0024 ±0.0023 ±0.0122 ±0.0140 ±0.0006 ±0.0007 ±0.0362 ±0.0394 ±0.0018 ±0.0021 ±0.0241 ±0.0310 ±0.0073 ±0.0076
0.079 0.225 0.098 0.250 0.320 0.476 0.118 0.275 0.433 0.567 0.123 0.280 0.184 0.349 0.199 0.356
1440
L ±0.0120 ±0.0158 ±0.0051 ±0.0066 ±0.0476 ±0.0436 ±0.0041 ±0.0045 ±0.2573 ±0.2044 ±0.0019 ±0.0019 ±0.0119 ±0.0139 ±0.0174 ±0.0193
0.074 0.215 0.143 0.303 0.539 0.637 0.183 0.355 0.629 0.698 0.187 0.359 0.334 0.504 0.356 0.486
2160
±0.0108 ±0.0166 ±0.0310 ±0.0373 ±0.0157 ±0.0107 ±0.0102 ±0.0116 ±0.1734 ±0.1050 ±0.0032 ±0.0039 ±0.0996 ±0.0985 ±0.0439 ±0.0235
0.124 0.269 0.139 0.290 0.122 0.264 0.136 0.292 0.128 0.271 0.129 0.275 0.192 0.343 0.129 0.277
96
S ±0.0043 ±0.0056 ±0.0030 ±0.0036 ±0.0013 ±0.0010 ±0.0019 ±0.0018 ±0.0008 ±0.0007 ±0.0014 ±0.0021 ±0.0198 ±0.0177 ±0.0046 ±0.0066
2hTTE
0.177 0.344 0.207 0.370 0.313 0.457 0.233 0.392 0.319 0.461 0.256 0.407 0.231 0.394 0.273 0.419
720 ±0.0393 ±0.0366 ±0.0113 ±0.0155 ±0.0048 ±0.0039 ±0.0046 ±0.0042 ±0.0215 ±0.0170 ±0.0041 ±0.0036 ±0.0155 ±0.0121 ±0.0132 ±0.0101
0.176 0.340 0.192 0.358 0.520 0.599 0.351 0.481 0.514 0.597 0.389 0.506 0.211 0.379 0.384 0.487
1440
L ±0.0042 ±0.0046 ±0.0574 ±0.0621 ±0.1879 ±0.1216 ±0.0172 ±0.0130 ±0.0591 ±0.0413 ±0.0081 ±0.0052 ±0.0165 ±0.0155 ±0.0327 ±0.0225
0.198 0.358 0.263 0.413 0.759 0.734 0.610 0.659 0.740 0.728 0.610 0.645 0.240 0.399 0.919 0.737
2160
±0.0367 ±0.0307 ±0.1384 ±0.1105 ±0.0469 ±0.0255 ±0.0552 ±0.0355 ±0.0932 ±0.0515 ±0.0120 ±0.0063 ±0.0284 ±0.0237 ±0.1815 ±0.0698
0.042 0.157 0.044 0.161 0.045 0.160 0.039 0.150 0.045 0.156 0.041 0.154 0.058 0.181 0.066 0.200
192
S ±0.0017 ±0.0061 ±0.0011 ±0.0014 ±0.0032 ±0.0060 ±0.0003 ±0.0005 ±0.0040 ±0.0062 ±0.0001 ±0.0003 ±0.0052 ±0.0079 ±0.0240 ±0.0402
1mTTE
0.090 0.238 0.085 0.227 0.099 0.245 0.091 0.237 0.105 0.252 0.098 0.247 0.142 0.298 0.093 0.238
1440
±0.0133 ±0.0258 ±0.0019 ±0.0022 ±0.0058 ±0.0060 ±0.0052 ±0.0076 ±0.0024 ±0.0028 ±0.0003 ±0.0002 ±0.0340 ±0.0350 ±0.0014 ±0.0013
0.089 0.238 0.092 0.242 0.137 0.294 0.096 0.245 0.175 0.344 0.101 0.251 0.181 0.339 0.127 0.282
2880
L ±0.0052 ±0.0066 ±0.0056 ±0.0071 ±0.0213 ±0.0252 ±0.0054 ±0.0075 ±0.0098 ±0.0110 ±0.0010 ±0.0011 ±0.1038 ±0.1067 ±0.0068 ±0.0083
0.083 0.228 0.090 0.241 0.247 0.412 0.110 0.265 0.271 0.437 0.119 0.275 0.167 0.324
4320 OOM OOM
±0.0094 ±0.0097 ±0.0033 ±0.0038 ±0.0516 ±0.0538 ±0.0048 ±0.0057 ±0.0111 ±0.0114 ±0.0003 ±0.0003 ±0.0113 ±0.0083
0.093 0.227 0.102 0.240 0.095 0.230 0.094 0.231 0.093 0.230 0.096 0.232 0.121 0.260 0.110 0.257
192
S ±0.0019 ±0.0027 ±0.0019 ±0.0028 ±0.0022 ±0.0025 ±0.0015 ±0.0018 ±0.0001 ±0.0001 ±0.0009 ±0.0015 ±0.0100 ±0.0084 ±0.0140 ±0.0184
2mTTE
0.215 0.362 0.228 0.378 0.243 0.388 0.226 0.380 0.237 0.384 0.235 0.386 0.280 0.424 0.264 0.408
1440
±0.0072 ±0.0055 ±0.0073 ±0.0067 ±0.0168 ±0.0123 ±0.0102 ±0.0103 ±0.0018 ±0.0010 ±0.0013 ±0.0026 ±0.0231 ±0.0152 ±0.0265 ±0.0212
0.211 0.370 0.236 0.391 0.322 0.465 0.243 0.396 0.322 0.464 0.262 0.412 0.268 0.417 0.302 0.441
2880
L ±0.0188 ±0.0170 ±0.0097 ±0.0103 ±0.0088 ±0.0066 ±0.0158 ±0.0136 ±0.0016 ±0.0014 ±0.0023 ±0.0020 ±0.0452 ±0.0297 ±0.0143 ±0.0108
0.215 0.376 0.234 0.393 0.448 0.555 0.307 0.450 0.448 0.553 0.333 0.468 0.409 0.518
4320 OOM OOM
±0.0242 ±0.0228 ±0.0295 ±0.0249 ±0.0335 ±0.0236 ±0.0157 ±0.0100 ±0.0182 ±0.0135 ±0.0047 ±0.0030 ±0.0312 ±0.0191
0.196 0.313 0.286 0.386 0.241 0.367 0.227 0.336 0.207 0.322 0.394 0.451 0.332 0.426 0.279 0.393
96
S ±0.0024 ±0.0043 ±0.0286 ±0.0188 ±0.0053 ±0.0086 ±0.0142 ±0.0102 ±0.0025 ±0.0021 ±0.0019 ±0.0021 ±0.0341 ±0.0239 ±0.0160 ±0.0102
yticirtcelE
0.275 0.386 0.417 0.471 0.336 0.446 0.332 0.426 0.304 0.412 0.467 0.504 0.505 0.533 0.417 0.486
720
±0.0471 ±0.0426 ±0.0270 ±0.0149 ±0.0732 ±0.0592 ±0.0015 ±0.0005 ±0.0077 ±0.0064 ±0.0032 ±0.0016 ±0.1046 ±0.0581 ±0.0319 ±0.0221
0.338 0.441 0.491 0.523 0.419 0.504 0.482 0.537 0.395 0.484 0.625 0.610 0.577 0.574 0.651 0.609
1440
L ±0.0256 ±0.0047 ±0.0245 ±0.0117 ±0.0346 ±0.0258 ±0.0046 ±0.0093 ±0.0150 ±0.0117 ±0.0028 ±0.0010 ±0.0591 ±0.0248 ±0.0470 ±0.0237
0.380 0.481 0.536 0.547 0.421 0.501 0.768 0.644 0.415 0.496 0.938 0.758 0.642 0.610 0.896 0.714
2160
±0.0307 ±0.0199 ±0.0526 ±0.0272 ±0.0190 ±0.0137 ±0.1452 ±0.0578 ±0.0098 ±0.0094 ±0.0039 ±0.0024 ±0.1659 ±0.0910 ±0.1156 ±0.0529
0.132 0.206 0.145 0.219 0.168 0.256 0.192 0.296 0.219 0.327 0.264 0.334 0.247 0.326 0.220 0.312
96
S ±0.0014 ±0.0008 ±0.0006 ±0.0013 ±0.0115 ±0.0113 ±0.0033 ±0.0020 ±0.0013 ±0.0016 ±0.0018 ±0.0021 ±0.0099 ±0.0086 ±0.0224 ±0.0184
cfifarT
0.144 0.225 0.163 0.269 0.304 0.394 0.213 0.318 0.309 0.419 0.247 0.329 0.277 0.360 0.255 0.344
720 ±0.0005 ±0.0006 ±0.0006 ±0.0002 ±0.0130 ±0.0124 ±0.0085 ±0.0079 ±0.0002 ±0.0002 ±0.0015 ±0.0014 ±0.0243 ±0.0235 ±0.0649 ±0.0546
0.174 0.251 0.188 0.292 0.375 0.443 0.246 0.341 0.353 0.409 0.311 0.390 0.303 0.361 0.297 0.376
1440
L ±0.0009 ±0.0016 ±0.0002 ±0.0219 ±0.0301 ±0.0250 ±0.0147 ±0.0143 ±0.0108 ±0.0083 ±0.0022 ±0.0025 ±0.0327 ±0.0204 ±0.0446 ±0.0358
0.175 0.252 0.190 0.304 0.360 0.426 0.261 0.353 0.324 0.386 0.988 0.745 0.222 0.317 0.317 0.394
2160
±0.0074 ±0.0082 ±0.0006 ±0.0171 ±0.0257 ±0.0043 ±0.0078 ±0.0053 ±0.0078 ±0.0061 ±0.0033 ±0.0021 ±0.0208 ±0.0288 ±0.0431 ±0.0419
0.521 0.522 0.584 0.536 0.569 0.525 0.545 0.539 0.579 0.529 0.589 0.533 0.636 0.567 0.703 0.625
96
S ±0.0522 ±0.0582 ±0.0114 ±0.0060 ±0.0102 ±0.0040 ±0.0012 ±0.0014 ±0.0074 ±0.0029 ±0.0034 ±0.0017 ±0.0227 ±0.0035 ±0.1516 ±0.0994
rehtaeW
0.963 0.715 1.090 0.753 1.080 0.754 0.987 0.752 1.007 0.706 1.003 0.728 1.007 0.725 1.114 0.822
720
±0.0193 ±0.0076 ±0.0297 ±0.0083 ±0.0500 ±0.0195 ±0.0133 ±0.0051 ±0.0293 ±0.0101 ±0.0046 ±0.0017 ±0.0350 ±0.0127 ±0.0325 ±0.0163
1.280 0.835 1.547 0.926 1.351 0.863 1.342 0.860 1.299 0.823 1.472 0.900 1.394 0.867 1.435 0.919
1440
L ±0.1150 ±0.0400 ±0.3281 ±0.0858 ±0.0199 ±0.0075 ±0.0160 ±0.0046 ±0.0248 ±0.0091 ±0.0030 ±0.0011 ±0.0730 ±0.0272 ±0.1262 ±0.0537
1.415 0.887 1.744 0.994 1.544 0.937 1.506 0.924 1.454 0.887 1.712 0.988 1.598 0.944 1.786 1.054
2160
±0.1449 ±0.0501 ±0.1810 ±0.0737 ±0.0482 ±0.0185 ±0.0333 ±0.0109 ±0.0160 ±0.0059 ±0.0027 ±0.0011 ±0.2034 ±0.0783 ±0.3309 ±0.1121
0.051 0.172 0.054 0.178 0.054 0.181 0.068 0.197 0.049 0.170 0.052 0.173 0.054 0.178 0.059 0.184
48
S ±0.0007 ±0.0019 ±0.0010 ±0.0019 ±0.0007 ±0.0010 ±0.0037 ±0.0059 ±0.0001 ±0.0014 ±0.0005 ±0.0011 ±0.0017 ±0.0038 ±0.0030 ±0.0041
egnahcxE
0.448 0.527 0.479 0.532 0.459 0.536 0.548 0.573 0.485 0.531 0.492 0.534 0.493 0.541 0.528 0.556
360
±0.0579 ±0.0141 ±0.0097 ±0.0062 ±0.0496 ±0.0208 ±0.0051 ±0.0056 ±0.0808 ±0.0561 ±0.0057 ±0.0034 ±0.0948 ±0.0407 ±0.0206 ±0.0082
1.067 0.794 1.239 0.856 1.383 0.927 1.264 0.859 1.718 1.024 1.291 0.864 1.358 0.894 1.381 0.903
720
L ±0.4944 ±0.1552 ±0.0556 ±0.0224 ±0.1069 ±0.0390 ±0.0369 ±0.0128 ±0.5115 ±0.1862 ±0.0167 ±0.0038 ±0.0793 ±0.0291 ±0.1234 ±0.0415
1.004 0.792 1.327 0.900 4.874 1.972 1.255 0.873 4.982 1.973 1.670 1.010 1.774 1.058 1.600 0.980
1080
±0.2767 ±0.0853 ±0.0578 ±0.0198 ±0.2948 ±0.0696 ±0.0269 ±0.0098 ±0.9254 ±0.1950 ±0.0821 ±0.0227 ±1.1516 ±0.3573 ±0.1925 ±0.0597
0.725 0.574 1.414 0.735 0.815 0.701 1.558 0.965 1.397 0.901 1.079 0.739 1.107 0.698 0.773 0.619
14
S ±0.0955 ±0.0467 ±0.2329 ±0.0204 ±0.0157 ±0.0088 ±0.1390 ±0.0428 ±0.1129 ±0.0420 ±0.0264 ±0.0126 ±0.1107 ±0.0399 ±0.0546 ±0.0341
ILI
0.887 0.683 1.604 0.854 1.670 1.062 1.878 1.110 2.008 1.134 1.315 0.887 1.515 0.767 0.989 0.770
28
±0.0756 ±0.0212 ±0.1568 ±0.0301 ±0.0739 ±0.0274 ±0.0419 ±0.0116 ±0.1461 ±0.0468 ±0.0776 ±0.0385 ±0.1455 ±0.0301 ±0.0787 ±0.0450
0.807 0.725 1.021 0.787 1.757 1.210 1.451 1.028 1.584 1.075 1.080 0.891 0.895 0.742 0.856 0.741
56
L ±0.0113 ±0.0118 ±0.0673 ±0.0307 ±0.0401 ±0.0129 ±0.0570 ±0.0150 ±0.0874 ±0.0300 ±0.0221 ±0.0122 ±0.0668 ±0.0550 ±0.0487 ±0.0286
1.499 1.038 1.669 1.072 3.593 1.759 2.846 1.438 3.332 1.572 2.608 1.387 1.724 1.108 1.660 1.097
112
±0.0908 ±0.0335 ±0.1123 ±0.0462 ±0.0515 ±0.0127 ±0.0670 ±0.0159 ±0.0821 ±0.0145 ±0.2469 ±0.0801 ±0.2964 ±0.1245 ±0.1745 ±0.0560
1stCount 52 2 2 2 4 1 0 0
23
