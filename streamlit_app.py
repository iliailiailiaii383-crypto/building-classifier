import io
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_REPO = "ILIAILIAII/zdanija-weights"
WEIGHTS_NAME = "zdanija.weights.h5"
IMG_SIZE = 224

LANGS = {
    "ru": {
        "page_title": "Гид по архитектуре Гомельского парка",
        "title": "🏛 Гид по архитектуре Гомельского парка",
        "description": (
            "Гомельский парк — уникальный ансамбль, где природа сочетается с величественной "
            "архитектурой. Дворец, соборы, служебные флигели и малые архитектурные формы "
            "создают неповторимый облик этого места.\n\n"
            "Это приложение разработано, чтобы стать вашим умным помощником в изучении этих "
            "построек. Ваша задача — гуляя по парку, идентифицировать архитектурные объекты, "
            "встречающиеся на вашем пути. Приложение поможет вам классифицировать их по стилю "
            "и эпохе, превращая обычную прогулку в увлекательное образовательное путешествие."
        ),
        "tab_upload": "📁 Загрузить фото",
        "tab_camera": "📷 Камера",
        "uploader": "Фото здания",
        "camera": "Сделать фото",
        "empty": "Здесь появится результат после загрузки фото.",
        "spinner": "Распознаю...",
        "error": "Ошибка обработки изображения: ",
        "expander": "Краткая информация",
        "names": [
            "Башня обозрения",
            "Дворец Румянцевых — Паскевичей",
            "Зимний сад",
            "Петропавловский собор",
            "Часовня-усыпальница князей Паскевичей",
        ],
        "info": [
            "\n\n".join([
                "Смотровая башня в парке Гомельского дворцово-паркового ансамбля — уникальный исторический объект, изначально построенный как промышленная труба, а сегодня являющийся самой высокой смотровой площадкой в городе.",
                "Изначально башня была вытяжной трубой сахарного завода, основанного в 1840-х годах в имении князя Ивана Паскевича.",
                "После переноса производства в Добруш в начале 1880-х годов старые заводские постройки приспособили под парковые нужды: наземный корпус стал Зимним садом, а труба — смотровой башней.",
                "Высота башни составляет от 40 до 42 метров, включая шпиль и смотровую площадку.",
                "Наверх ведет узкая винтовая лестница, после реконструкции количество ступеней составляет около 204 (по другим данным — 195).",
                "В годы Великой Отечественной войны сооружение пострадало от обстрелов, следы от осколков сохранились на кирпичной кладке.",
                "С начала 2000-х годов башня была закрыта из-за аварийного состояния лестниц.",
                "Реконструкция началась в 2015 году и обошлась примерно в 1,5 млрд рублей из областного бюджета.",
                "Обновленный объект торжественно открыли 26 ноября 2015 года.",
                "В ходе работ заменили металлические конструкции, укрепили кладку, установили стеклянный купол на площадке и современную подсветку.",
                "С площадки открывается панорамный вид на дворец Румянцевых-Паскевичей, старинный парк, реку Сож, набережную, Новобелицкий район и купола Петропавловского собора.",
                "Башня расположена в южной части парка Гомельского дворцово-паркового ансамбля, недалеко от Зимнего сада, в Центральном районе города.",
                "Режим работы — обычно с 10:00 до 18:00, продажа билетов прекращается за 30 минут до закрытия, выходной день — понедельник.",
            ]),
            "\n\n".join([
                "Дворец Румянцевых и Паскевичей — главный архитектурный памятник Гомеля и композиционный центр дворцово-паркового ансамбля, построенный в стиле классицизма во второй половине XVIII века.",
                "История дворца начинается в 1775 году, когда императрица Екатерина II подарила Гомель генерал-фельдмаршалу Петру Александровичу Румянцеву-Задунайскому за победы в войне с Турцией.",
                "На месте старого деревянного замка князя Чарторыйского в 1777 году началось строительство новой резиденции.",
                "Автором проекта предположительно был архитектор Иван Старов, создатель Таврического дворца в Санкт-Петербурге.",
                "Строительство главного корпуса завершилось в 1796 году.",
                "После смерти Петра Румянцева имение унаследовал его сын Николай Петрович Румянцев — канцлер Российского государства, меценат и покровитель наук.",
                "В 1834 году новым владельцем Гомеля стал генерал-фельдмаршал Иван Фёдорович Паскевич, при котором дворец был реконструирован под руководством архитектора Адама Идзковского.",
                "Паскевич также создал вокруг дворца пейзажный парк с экзотическими растениями, беседками, фонтанами и скульптурами.",
                "В XIX — начале XX века во дворце устраивались пышные приёмы и балы, здесь неоднократно бывали члены императорской семьи Романовых.",
                "Дворец сильно пострадал во время Стрекопытовского мятежа в 1919 году и в годы Великой Отечественной войны.",
                "Восстановительные работы после войны завершились в 1960-е годы.",
                "С 1919 года в стенах дворца работает музей, основой которого стали коллекции, собранные Румянцевыми и Паскевичами.",
                "Сегодня на первом этаже центральной части дворца воссозданы исторические интерьеры: Колонный зал, Белая гостиная, Красная гостиная и зал торжественных приёмов.",
                "В этих залах проходят выставки, концерты, балы и встречи на высшем уровне.",
                "Музейное собрание включает богатейшие коллекции живописи, икон, оружия, нумизматики, старопечатных книг и мемориальные комплексы выдающихся людей.",
            ]),
            "\n\n".join([
                "Зимний сад в Гомеле — это не просто оранжерея, а памятник архитектуры XIX века и одна из самых посещаемых достопримечательностей дворцово-паркового ансамбля.",
                "Здание, где сейчас находится сад, имеет промышленное прошлое.",
                "Оно было построено в первой половине XIX века как один из корпусов сахарного завода князя Паскевича.",
                "В 1877 году князь Фёдор Иванович Паскевич распорядился переоборудовать бывший цех в оранжерею.",
                "Для поддержания тепла зимой использовались две большие печи в подвале.",
                "Стены изнутри выложили природными минералами — это позволяло лазающим растениям создавать живой зелёный ковёр.",
                "Площадь сада составляет около 245 квадратных метров.",
                "Здесь собрано более 600 экземпляров растений, представляющих порядка 18–26 видов.",
                "Среди них магнолия, различные пальмы, кофейное дерево, инжир, лимон, авокадо, лавр благородный, жасмин гималайский, плющ колхидский, папоротники, аспидистра высокая, тисс ягодный и другие.",
                "Самое старое растение — вееролистная пальма, посаженная в 1888 году.",
                "Зимний сад — это ещё и небольшой живой уголок.",
                "Здесь живут декоративные кролики, морские свинки, черепахи и разноцветные аквариумные рыбки.",
                "Именно появление интерактивного мини-зоопарка сделало это место самым посещаемым объектом ансамбля — ежемесячно сюда приходят более 1 тыс. человек.",
                "Адрес: город Гомель, площадь Ленина, 4, корпус 5.",
                "Режим работы: среда — воскресенье с 10:00 до 18:00, вход до 17:45, перерыв 13:30–14:00.",
                "Выходные дни: понедельник и вторник.",
                "Стоимость: взрослый билет — 11,00 руб., для пенсионеров и учащихся — 5,50 руб.",
            ]),
            "\n\n".join([
                "Петропавловский кафедральный собор — главная православная святыня Гомеля и один из ключевых объектов дворцово-паркового ансамбля.",
                "Собор расположен на высоком мысу над рекой Сож, рядом с дворцом Румянцевых и Паскевичей.",
                "Инициатором строительства выступил граф Николай Петрович Румянцев — владелец Гомеля в начале XIX века.",
                "В марте 1808 года Румянцев обратился к архиепископу Могилевскому с просьбой разрешить построить в Гомеле каменную церковь.",
                "Автором проекта стал английский архитектор Джон Кларк.",
                "Закладка храма состоялась 18 октября 1809 года.",
                "Строительство шло с перерывами: его прервала война с Наполеоном 1812 года.",
                "Возведение завершилось в 1819 году, а освящение состоялось в мае 1824 или 1825 года.",
                "Собор построен в стиле классицизма и имеет высоту 25 метров.",
                "В плане здание представляет собой вытянутый крест с куполом на световом барабане.",
                "Фасады украшают четыре шестиколонных портика дорического ордера.",
                "Граф Румянцев завещал похоронить себя в этом соборе, что и было исполнено в 1826 году.",
                "В 1872 году церковь получила статус собора, а в 1907 году стала кафедральным.",
                "В советские годы собор закрывался: в 1935 году здесь разместился исторический музей.",
                "С 1962 по 1985 год в здании работал планетарий.",
                "В 1989 году собор был возвращён верующим.",
                "Сегодня в соборе хранятся православные святыни: частицы мощей святителя Николая Чудотворца, пророка Иоанна Крестителя, Святой Крест с частицей Животворящего Креста Господня и мощи преподобной Манефы Гомельской.",
            ]),
            "\n\n".join([
                "Часовня-усыпальница князей Паскевичей — памятник архитектуры и усыпальница рода Паскевичей, расположенная в Гомельском дворцово-парковом ансамбле.",
                "Часовня построена в 1870-х годах по проекту академика архитектуры Евгения Червинского.",
                "Инициатором строительства выступил сын полководца Ивана Паскевича — Фёдор Иванович Паскевич.",
                "Здание возведено в псевдорусском стиле и напоминает небольшой храм.",
                "Часовня сложена из кирпича и имеет шатровое завершение с главкой.",
                "Внутри находится мозаичное панно и мраморный иконостас.",
                "Усыпальница служила местом захоронения представителей рода Паскевичей.",
                "Здесь похоронены сам Иван Фёдорович Паскевич, его жена Елизавета Алексеевна и их сын Фёдор Иванович.",
                "В 1889 году в часовне был установлен мраморный саркофаг для Фёдора Ивановича Паскевича.",
                "В советские годы часовня использовалась не по назначению, внутреннее убранство было утрачено.",
                "В 2003 году началась реставрация часовни.",
                "После восстановления в 2012 году она вновь открыта для посещения.",
                "Сегодня часовня-усыпальница является одним из красивейших сооружений парка и популярным местом среди туристов.",
            ]),
        ],
    },
    "be": {
        "page_title": "Гід па архітэктуры Гомельскага парка",
        "title": "🏛 Гід па архітэктуры Гомельскага парка",
        "description": (
            "Гомельскі парк — унікальны ансамбль, дзе прырода спалучаецца з велічнай "
            "архітэктурай. Палац, саборы, службовыя флігелі і малыя архітэктурныя формы "
            "ствараюць непаўторны аблічча гэтага месца.\n\n"
            "Гэта прыкладанне створана, каб стаць вашым разумным памочнікам у вывучэнні гэтых "
            "пабудоў. Ваша задача — гуляючы па парку, ідэнтыфікаваць архітэктурныя аб'екты, "
            "якія сустракаюцца на вашым шляху. Прыкладанне дапамажа вам класіфікаваць іх па "
            "стылі і эпохі, ператвараючы звычайную прагулку ў захапляльнае адукацыйнае падарожжа."
        ),
        "tab_upload": "📁 Загрузіць фота",
        "tab_camera": "📷 Камера",
        "uploader": "Фота будынка",
        "camera": "Зрабіць фота",
        "empty": "Тут з'явіцца вынік пасля загрузкі фота.",
        "spinner": "Распазнаю...",
        "error": "Памылка апрацоўкі выявы: ",
        "expander": "Кароткая інфармацыя",
        "names": [
            "Аглядная вежа",
            "Палац Румянцеваў — Паскевічаў",
            "Зімовы сад",
            "Пятра-Паўлаўскі сабор",
            "Капліца-пахаванне князёў Паскевічаў",
        ],
        "info": [
            "\n\n".join([
                "Аглядная вежа ў парку Гомельскага палацава-паркавага ансамбля — унікальны гістарычны аб'ект, першапачаткова пабудаваны як прамысловая труба, а сёння якая з'яўляецца самай высокай агляднай пляцоўкай у горадзе.",
                "Ізначальна вежа была выцяжнай трубай цукровага завода, заснаванага ў 1840-я гады ў маёнтку князя Івана Паскевіча.",
                "Пасля перанясення вытворчасці ў Добруш у пачатку 1880-х гадоў старыя заводскія пабудовы прыстасавалі пад патрэбы парку: наземны корпус стаў Зімовым садам, а труба — агляднай вежай.",
                "Вышыня вежы складае ад 40 да 42 метраў, уключаючы шпіль і аглядную пляцоўку.",
                "Наверх вядзе вузкая вінтавая лесвіца, пасля рэканструкцыі колькасць прыступкаў складае каля 204 (па іншых дадзеных — 195).",
                "У гады Вялікай Айчыннай вайны збудаванне пацярпела ад абстрэлаў, сляды ад асколкаў захаваліся на цагляной кладцы.",
                "З пачатку 2000-х гадоў вежа была закрыта з-за аварыйнага стану лесвіц.",
                "Рэканструкцыя пачалася ў 2015 годзе і каштавала каля 1,5 млрд рублёў з абласнога бюджэту.",
                "Аноўлены аб'ект урачыста адкрылі 26 лістапада 2015 года.",
                "У ходзе работ замянілі металічныя канструкцыі, умацавалі кладку, усталявалі шкляны купол на пляцоўцы і сучаснае падсвятленне.",
                "З пляцоўкі адкрываецца панарамны від на палац Румянцевых-Паскевічаў, старадаўні парк, раку Сож, набярэжную, Навабеліцкі раён і куполы Пятра-Паўлаўскага сабора.",
                "Вежа размешчана ў паўднёвай частцы парку Гомельскага палацава-паркавага ансамбля, недалёка ад Зімовага сада, у Цэнтральным раёне горада.",
                "Рэжым працы — звычайна з 10:00 да 18:00, продаж білетоў спыняецца за 30 хвілін да закрыцця, выхадны дзень — панядзелак.",
            ]),
            "\n\n".join([
                "Палац Румянцевых і Паскевічаў — галоўны архітэктурны помнік Гомеля і кампазіцыйны цэнтр палацава-паркавага ансамбля, пабудаваны ў стылі класіцызму ў другой палове XVIII стагоддзя.",
                "Гісторыя палаца пачынаецца ў 1775 годзе, калі імператрыца Кацярына II падарыла Гомель генерал-фельдмаршалу Пятру Аляксандравічу Румянцеву-Задунайскаму за перамогі ў вайне з Турцыяй.",
                "На месцы старога драўлянага замка князя Чартарыйскага ў 1777 годзе пачалося будаўніцтва новай рэзідэнцыі.",
                "Аўтарам праекта меркавана быў архітэктар Іван Староў, стваральнік Таўрычнага палаца ў Санкт-Пецярбургу.",
                "Будаўніцтва галоўнага корпуса завяршылася ў 1796 годзе.",
                "Пасля смерці Пятра Румянцева маёнтак успадкаваў яго сын Мікалай Пятровіч Румянцаў — канцлер Расійскай дзяржавы, мецэнат і заступнік навук.",
                "У 1834 годзе новым уладальнікам Гомеля стаў генерал-фельдмаршал Іван Фёдаравіч Паскевіч, пры якім палац быў рэканструяваны пад кіраўніцтвам архітэктара Адама Ідзкоўскага.",
                "Паскевіч таксама стварыў вакол палаца пейзажны парк з экзатычнымі раслінамі, беседкамі, фантанамі і скульптурамі.",
                "У XIX — пачатку XX стагоддзя ў палацы ўладкоўваліся пышныя прыёмы і балі, тут неаднаразова бывалі члены імператарскай сям'і Раманавых.",
                "Палац моцна пацярпеў падчас Стрэкацоваўскага паўстання ў 1919 годзе і ў гады Вялікай Айчыннай вайны.",
                "Аднаўленчыя работы пасля вайны завяршыліся ў 1960-я гады.",
                "З 1919 года ў сценах палаца працуе музей, асновай якога сталі калекцыі, сабраныя Румянцевымі і Паскевічамі.",
                "Сёння на першым паверсе цэнтральнай часткі палаца адноўлены гістарычныя інтэр'еры: Калонная зала, Белая гасціная, Чырвоная гасціная і зала ўрачыстых прыёмаў.",
                "У гэтых залах праходзяць выставы, канцэрты, балі і сустрэчы на вышэйшым узроўні.",
                "Музейная калекцыя ўключае багатыя калекцыі жывапісу, іканаграфіі, зброі, нумізматыкі, старадрукаваных кніг і мемарыяльныя комплексы выдатных людзей.",
            ]),
            "\n\n".join([
                "Зімовы сад у Гомелі — гэта не проста аранжэрыя, а помнік архітэктуры XIX стагоддзя і адна з самых наведвальных славутасцей палацава-паркавага ансамбля.",
                "Будынак, дзе цяпер знаходзіцца сад, мае прамысловае мінулае.",
                "Ён быў пабудаваны ў першай палове XIX стагоддзя як адзін з карпусоў цукровага завода князя Паскевіча.",
                "У 1877 годзе князь Фёдар Іванавіч Паскевіч распарадзіўся пераабсталяваць былы цэх у аранжэрыю.",
                "Для падтрымання цяпла зімой выкарыстоўваліся дзве вялікія печы ў падвале.",
                "Сцены знутры выкладзены прыроднымі мінераламі — гэта дазваляла лазячым раслінам ствараць жывы зялёны дыван.",
                "Плошча сада складае каля 245 квадратных метраў.",
                "Тут сабрана больш за 600 экзэмпляраў раслін, якія прадстаўляюць каля 18–26 відаў.",
                "Сярод іх магнолія, розныя пальмы, кававае дрэва, смакоўніца, лімон, авакада, лаўр шляхетны, жасмен гімалайскі, плюшч калхідскі, папараці, аспідістра высокая, ціс ягадны і іншыя.",
                "Самая старая расліна — веералісцістая пальма, пасаджаная ў 1888 годзе.",
                "Зімовы сад — гэта яшчэ і невялікі жывы куток.",
                "Тут жывуць дэкаратыўныя трусы, марскія свінкі, чарапахі і разнаколерныя акварыумныя рыбкі.",
                "Менавіта з'яўленне інтэрактыўнага міні-заапарка зрабіла гэта месца самым наведвальным аб'ектам ансамбля — штомесяц сюды прыходзяць больш за 1 тыс. чалавек.",
                "Адрас: горад Гомель, плошча Леніна, 4, корпус 5.",
                "Рэжым працы: серада — нядзеля з 10:00 да 18:00, уваход да 17:45, перапынак 13:30–14:00.",
                "Выхадныя дні: панядзелак і аўторак.",
                "Кошт: дарослы білет — 11,00 руб., для пенсіянераў і навучэнцаў — 5,50 руб.",
            ]),
            "\n\n".join([
                "Пятра-Паўлаўскі кафедральны сабор — галоўная праваслаўная святыня Гомеля і адзін з ключавых аб'ектаў палацава-паркавага ансамбля.",
                "Сабор размешчаны на высокім мысе над ракой Сож, побач з палацам Румянцевых і Паскевічаў.",
                "Ініцыятарам будаўніцтва выступіў граф Мікалай Пятровіч Румянцаў — уладальнік Гомеля ў пачатку XIX стагоддзя.",
                "У сакавіку 1808 года Румянцаў звярнуўся да архібіскупа Магілёўскага з просьбай дазволіць пабудаваць у Гомелі каменную царкву.",
                "Аўтарам праекта стаў англійскі архітэктар Джон Кларк.",
                "Закладка храма адбылася 18 кастрычніка 1809 года.",
                "Будаўніцтва ішло з перапынкамі: яно было перапынена вайной з Напалеонам у 1812 годзе.",
                "Узвядзенне завяршылася ў 1819 годзе, а асвячэнне адбылося ў маі 1824 або 1825 года.",
                "Сабор пабудаваны ў стылі класіцызму і мае вышыню 25 метраў.",
                "У плане будынак уяўляе сабой выцягнуты крыж з куполам на святлавым барабане.",
                "Фасады ўпрыгожваюць чатыры порцікі з шасцю слупамі дарычнага ордэра.",
                "Граф Румянцаў запавядаў пахаваць сябе ў гэтым саборы, што і было выканана ў 1826 годзе.",
                "У 1872 годзе царква атрымала статус сабора, а ў 1907 годзе стала кафедральным.",
                "У савецкія гады сабор зачыняўся: у 1935 годзе тут размясціўся гістарычны музей.",
                "З 1962 па 1985 год у будынку працавалі планетарый.",
                "У 1989 годзе сабор быў вернуты вернікам.",
                "Сёння ў саборы захоўваюцца праваслаўныя святыні: часціцы мошчаў святога Мікалая Цудатворца, прарока Іаана Хрысціцеля, Святы Крыж з часткай Жыватворнага Крыжа Гасподняга і мошчы прападобнай Манефы Гомельскай.",
            ]),
            "\n\n".join([
                "Капліца-пахаванне князёў Паскевічаў — помнік архітэктуры і пахаванне роду Паскевічаў, размешчанае ў Гомельскім палацава-паркавым ансамблі.",
                "Капліца пабудавана ў 1870-я гады па праекце акадэміка архітэктуры Яўгена Чэрвінскага.",
                "Ініцыятарам будаўніцтва выступіў сын палкаводца Івана Паскевіча — Фёдар Іванавіч Паскевіч.",
                "Будынак узведзены ў псевдарускім стылі і нагадвае невялікі храм.",
                "Капліца складзена з цэглы і мае шатровае завяршэнне з галоўкай.",
                "Унутры знаходзіцца мазаічнае пано і мармуровы іканастас.",
                "Пахаванне служыла месцам пахавання прадстаўнікоў роду Паскевічаў.",
                "Тут пахаваны сам Іван Фёдаравіч Паскевіч, яго жонка Лізавета Аляксеяўна і іх сын Фёдар Іванавіч.",
                "У 1889 годзе ў капліцы быў усталяваны мармуровы саркафаг для Фёдара Іванавіча Паскевіча.",
                "У савецкія гады капліца выкарыстоўвалася не па прызначэнні, унутранае ўбранне было страчана.",
                "У 2003 годзе пачалася рэстаўрацыя капліцы.",
                "Пасля аднаўлення ў 2012 годзе яна зноў адкрыта для наведвання.",
                "Сёння капліца-пахаванне з'яўляецца адным з прыгажэйшых збудаванняў парка і папулярным месцам сярод турыстаў.",
            ]),
        ],
    },
    "en": {
        "page_title": "Architectural Guide to Gomel Park",
        "title": "🏛 Architectural Guide to Gomel Park",
        "description": (
            "Gomel Park is a unique ensemble where nature blends with majestic architecture. "
            "The palace, cathedrals, service wings and small architectural forms give the place "
            "its unmistakable character.\n\n"
            "This app is your smart companion for exploring these buildings. As you stroll "
            "through the park, identify the architectural landmarks you come across — the app "
            "will classify them by style and era, turning an ordinary walk into an engaging "
            "educational journey."
        ),
        "tab_upload": "📁 Upload photo",
        "tab_camera": "📷 Camera",
        "uploader": "Building photo",
        "camera": "Take a photo",
        "empty": "The result will appear here after you upload a photo.",
        "spinner": "Recognizing...",
        "error": "Image processing error: ",
        "expander": "Brief information",
        "names": [
            "Observation Tower",
            "Rumyantsev–Paskievich Palace",
            "Winter Garden",
            "Saints Peter and Paul Cathedral",
            "Chapel-Mausoleum of the Paskievich Princes",
        ],
        "info": [
            "\n\n".join([
                "The observation tower in the park of the Gomel palace and park ensemble is a unique historic structure, originally built as an industrial chimney and now the highest viewing platform in the city.",
                "The tower began life as the chimney of a sugar factory founded in the 1840s in the estate of Prince Ivan Paskievich.",
                "After production moved to Dobrush in the early 1880s, the old factory buildings were repurposed for the park: the ground-level building became the Winter Garden and the chimney became the observation tower.",
                "The tower stands 40 to 42 metres tall, including the spire and the viewing deck.",
                "A narrow spiral staircase leads to the top; after reconstruction it has about 204 steps (195 according to other sources).",
                "During the Great Patriotic War (World War II) the structure was damaged by shelling, and shrapnel marks remain in the brickwork.",
                "From the early 2000s the tower was closed because its staircases were in a dangerous condition.",
                "Reconstruction began in 2015 and cost around 1.5 billion rubles from the regional budget.",
                "The renewed attraction officially opened on 26 November 2015.",
                "The works replaced the steel structures, reinforced the masonry, and installed a glass dome on the platform as well as modern lighting.",
                "The platform offers a panoramic view of the Rumyantsev–Paskievich Palace, the old park, the Sozh River, the embankment, the Novobelitsky district and the domes of Saints Peter and Paul Cathedral.",
                "The tower stands in the southern part of the Gomel palace and park ensemble, close to the Winter Garden, in the city's Central District.",
                "Opening hours are usually 10:00 to 18:00; ticket sales stop 30 minutes before closing, and the tower is closed on Mondays.",
            ]),
            "\n\n".join([
                "The Palace of the Rumyantsevs and Paskievichs is the principal architectural landmark of Gomel and the compositional centre of the palace and park ensemble, built in the classical style in the second half of the 18th century.",
                "The palace's story begins in 1775, when Empress Catherine II presented Gomel to Field Marshal Pyotr Alexandrovich Rumyantsev-Zadunaysky in reward for his victories in the war against Turkey.",
                "Construction of a new residence on the site of the old wooden castle of Prince Czartoryski began in 1777.",
                "The project is believed to have been designed by architect Ivan Starov, creator of the Tauride Palace in St Petersburg.",
                "The main block was completed in 1796.",
                "After Pyotr Rumyantsev's death the estate passed to his son Nikolai Petrovich Rumyantsev — chancellor of the Russian state, patron of the arts and supporter of the sciences.",
                "In 1834 Gomel passed to Field Marshal Ivan Fyodorovich Paskievich, under whom the palace was reconstructed under architect Adam Idzkowski.",
                "Paskievich also laid out a landscape park around the palace, with exotic plants, pavilions, fountains and sculptures.",
                "In the 19th and early 20th centuries the palace hosted lavish receptions and balls, and members of the Romanov imperial family visited several times.",
                "The palace was badly damaged during the Strekoputov revolt of 1919 and during the Great Patriotic War.",
                "Post-war restoration was completed in the 1960s.",
                "A museum has worked within the palace walls since 1919, built on collections assembled by the Rumyantsevs and Paskievichs.",
                "Today the historic interiors of the central part of the palace's first floor have been recreated: the Colonnade Hall, the White Drawing Room, the Red Drawing Room and the Hall of Ceremonial Receptions.",
                "These halls host exhibitions, concerts, balls and high-level meetings.",
                "The museum collection includes rich holdings of painting, icons, weapons, numismatics, early printed books and memorial complexes dedicated to outstanding figures.",
            ]),
            "\n\n".join([
                "The Winter Garden in Gomel is not just a greenhouse: it is a 19th-century architectural monument and one of the most visited attractions of the palace and park ensemble.",
                "The building that now houses the garden has an industrial past.",
                "It was built in the first half of the 19th century as one of the blocks of Prince Paskievich's sugar factory.",
                "In 1877 Prince Fyodor Ivanovich Paskievich ordered the former workshop converted into a greenhouse.",
                "Two large stoves in the cellar were used to keep it warm in winter.",
                "The walls are lined inside with natural minerals, which allowed climbing plants to form a living green carpet.",
                "The garden covers about 245 square metres.",
                "It contains more than 600 plant specimens representing some 18–26 species.",
                "These include magnolia, various palms, the coffee tree, fig, lemon, avocado, sweet bay, Himalayan jasmine, Colchic ivy, ferns, tall aspidistra, the yew and others.",
                "The oldest plant is a fan palm planted in 1888.",
                "The Winter Garden is also a small living corner.",
                "It is home to decorative rabbits, guinea pigs, turtles and colourful aquarium fish.",
                "It was the arrival of this interactive mini-zoo that made the garden the most visited site of the ensemble — more than 1,000 people come every month.",
                "Address: Gomel, Lenina Square 4, building 5.",
                "Opening hours: Wednesday–Sunday 10:00–18:00, last entry 17:45, break 13:30–14:00.",
                "Closed on Mondays and Tuesdays.",
                "Admission: adult ticket 11.00 rubles; for pensioners and students 5.50 rubles.",
            ]),
            "\n\n".join([
                "Saints Peter and Paul Cathedral is the main Orthodox shrine of Gomel and one of the key sites of the palace and park ensemble.",
                "The cathedral stands on a high bluff above the Sozh River, next to the Palace of the Rumyantsevs and Paskievichs.",
                "Its initiator was Count Nikolai Petrovich Rumyantsev, owner of Gomel in the early 19th century.",
                "In March 1808 Rumyantsev appealed to the Archbishop of Mogilev for permission to build a stone church in Gomel.",
                "The project was designed by the English architect John Clark.",
                "The foundation stone was laid on 18 October 1809.",
                "Construction proceeded in fits and starts — it was interrupted by the war against Napoleon in 1812.",
                "The building was completed in 1819 and consecrated in May 1824 or 1825.",
                "The cathedral was built in the classical style and stands 25 metres tall.",
                "In plan it is an elongated cross crowned by a dome on a drum with windows.",
                "The facades are decorated with four six-columned Doric porticos.",
                "Count Rumyantsev was buried in this cathedral by his own will, which was carried out in 1826.",
                "The church gained cathedral status in 1872 and became a cathedral church in 1907.",
                "In Soviet times the cathedral was closed — a history museum occupied it from 1935.",
                "A planetarium operated in the building from 1962 to 1985.",
                "The cathedral was returned to the faithful in 1989.",
                "Today it holds Orthodox shrines: relics of Saint Nicholas the Wonderworker, John the Baptist, a particle of the True Cross and the relics of Saint Manefa of Gomel.",
            ]),
            "\n\n".join([
                "The Chapel-Mausoleum of the Paskievich Princes is an architectural monument and the burial place of the Paskievich family, located in the Gomel palace and park ensemble.",
                "The chapel was built in the 1870s to a design by academician of architecture Yevgeny Chervinsky.",
                "Its initiator was the son of the commander Ivan Paskievich, Fyodor Ivanovich Paskievich.",
                "The building was erected in the pseudo-Russian style and resembles a small church.",
                "The chapel is made of brick and crowned with a tented roof and a small dome.",
                "Inside there is a mosaic panel and a marble iconostasis.",
                "The mausoleum served as the burial place of the Paskievich family.",
                "Ivan Fyodorovich Paskievich himself, his wife Yelizaveta Alekseyevna and their son Fyodor Ivanovich are buried here.",
                "In 1889 a marble sarcophagus for Fyodor Ivanovich Paskievich was installed in the chapel.",
                "In Soviet times the chapel was used for other purposes and its interior decoration was lost.",
                "Restoration of the chapel began in 2003.",
                "After restoration in 2012 it reopened to visitors.",
                "Today the chapel-mausoleum is one of the park's most beautiful structures and a popular spot for tourists.",
            ]),
        ],
    },
}

LANG_LABELS = {
    "ru": "🇷🇺 Русский",
    "be": "🇧🇾 Беларуская",
    "en": "🇬🇧 English",
}

st.set_page_config(
    page_title=LANGS[st.session_state.get("lang", "ru")]["page_title"],
    page_icon="🏛",
    layout="centered",
)

MOBILE_CSS = """
<style>
.block-container {max-width: 760px; padding-top: 1rem; padding-bottom: 1rem;}
h1 {font-size: 1.7rem !important;}
h1, h2, h3 {font-family: inherit;}
[data-testid="stButton"] button {min-height: 46px; width: 100%; font-size: 1rem;}
[data-testid="stFileUploaderDropzone"] {min-height: 90px;}
[data-testid="stProgress"] > div > div > div > div {height: 14px;}
@media (max-width: 640px) {
  .block-container {padding-left: 0.6rem; padding-right: 0.6rem;}
  h1 {font-size: 1.35rem !important;}
}
</style>
"""


@st.cache_resource
def get_weights_path() -> str:
    for candidate in (
        os.path.join(BASE_DIR, WEIGHTS_NAME),
        os.path.join(BASE_DIR, "..", WEIGHTS_NAME),
    ):
        if os.path.exists(candidate):
            return candidate
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=WEIGHTS_REPO, filename=WEIGHTS_NAME)


@st.cache_resource
def load_model():
    base = tf.keras.applications.VGG19(
        weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    for layer in base.layers:
        layer.trainable = False
    x = tf.keras.layers.Flatten()(base.output)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=predictions)
    model.load_weights(get_weights_path())
    return model


@st.cache_data
def predict(image_bytes: bytes):
    model = load_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    preds = model.predict(arr[np.newaxis, ...], verbose=0)[0]
    return [float(p) for p in preds]


st.markdown(MOBILE_CSS, unsafe_allow_html=True)

lang = st.radio(
    "Язык / Мова / Language",
    options=["ru", "be", "en"],
    format_func=lambda c: LANG_LABELS[c],
    key="lang",
    horizontal=True,
    label_visibility="collapsed",
)

L = LANGS[lang]

st.title(L["title"])
st.markdown(L["description"])

tab_upload, tab_camera = st.tabs([L["tab_upload"], L["tab_camera"]])
uploaded = tab_upload.file_uploader(
    L["uploader"], type=["jpg", "jpeg", "png", "bmp", "webp"], key="uploader"
)
camera = tab_camera.camera_input(L["camera"], key="camera")

img_bytes = None
if uploaded is not None:
    img_bytes = uploaded.getvalue()
elif camera is not None:
    img_bytes = camera.getvalue()

if img_bytes is None:
    st.info(L["empty"])
else:
    st.image(
        Image.open(io.BytesIO(img_bytes)).convert("RGB"), width="stretch"
    )

    with st.spinner(L["spinner"]):
        try:
            probs = predict(img_bytes)
        except Exception as e:
            st.error(L["error"] + str(e))
            probs = None

    if probs is not None:
        idx = int(np.argmax(probs))

        st.markdown(f"## {L['names'][idx]}")

        with st.expander(L["expander"], expanded=True):
            st.markdown(L["info"][idx])
