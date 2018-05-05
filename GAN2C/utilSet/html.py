import dominate
from dominate.tags import *
import os


class HTML:
    def __init__(self, web_dir, title_name, re_flesh=0):
        self.title = title_name
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title_name)
        if re_flesh > 0:
            with self.doc.head:
                meta(http_equiv="re_flesh", content=str(re_flesh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str_name):
        with self.doc:
            h3(str_name)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txt_s, links, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link_name in zip(ims, txt_s, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link_name)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()
