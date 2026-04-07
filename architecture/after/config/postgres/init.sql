CREATE TABLE public.misuratore (
	id int NOT NULL,
	nome varchar NOT NULL,
	CONSTRAINT misuratore_pk PRIMARY KEY (id)
);

INSERT INTO public.misuratore (id, nome) VALUES(7001, 'Meter 7001');
INSERT INTO public.misuratore (id, nome) VALUES(15805, 'Meter 15805');
INSERT INTO public.misuratore (id, nome) VALUES(18052, 'Meter 18052');
INSERT INTO public.misuratore (id, nome) VALUES(50176, 'Meter 50176');
INSERT INTO public.misuratore (id, nome) VALUES(115138, 'Meter 115138');

CREATE TABLE public.misura (
	misuratore_id int NOT NULL,
	"timestamp" timestamp with time zone NOT NULL,
	consumo float4 NULL,
	CONSTRAINT misura_misuratore_fk FOREIGN KEY (misuratore_id) REFERENCES public.misuratore(id)
);
CREATE UNIQUE INDEX misura_misuratore_id_idx ON public.misura (misuratore_id,"timestamp");
CREATE UNIQUE INDEX misura_timestamp_idx ON public.misura ("timestamp",misuratore_id);

CREATE TABLE public.modello (
	id int GENERATED ALWAYS AS IDENTITY NOT NULL,
	nome varchar NOT NULL,
	CONSTRAINT modello_pk PRIMARY KEY (id)
);

INSERT INTO public.modello (nome) VALUES ('modello-cnn3d') 

CREATE TABLE public.previsione (
	misuratore_id int NOT NULL,
	modello_id int NOT NULL,
	"timestamp" timestamp with time zone NOT NULL,
	id int GENERATED ALWAYS AS IDENTITY NOT NULL,
	CONSTRAINT previsione_pk PRIMARY KEY (id),
	CONSTRAINT previsione_misuratore_fk FOREIGN KEY (misuratore_id) REFERENCES public.misuratore(id),
	CONSTRAINT previsione_modello_fk FOREIGN KEY (modello_id) REFERENCES public.modello(id)
);
CREATE UNIQUE INDEX previsione_modello_id_idx ON public.previsione (modello_id,misuratore_id,"timestamp");

CREATE TABLE public.step (
	id_previsione int NOT NULL,
	"timestamp" timestamp with time zone NOT NULL,
	valore float4 NULL,
	CONSTRAINT step_previsione_fk FOREIGN KEY (id_previsione) REFERENCES public.previsione(id)
);

CREATE ROLE nodered NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT LOGIN NOREPLICATION NOBYPASSRLS PASSWORD 'nodered';
GRANT INSERT ON TABLE public.misura TO nodered;
GRANT SELECT ON TABLE public.misuratore TO nodered;

CREATE ROLE grafana NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT LOGIN NOREPLICATION NOBYPASSRLS PASSWORD 'grafana';
GRANT SELECT ON TABLE public.misura TO grafana;
GRANT SELECT ON TABLE public.misuratore TO grafana;
GRANT SELECT ON TABLE public.modello TO grafana;
GRANT SELECT ON TABLE public.previsione TO grafana;
GRANT SELECT ON TABLE public.step TO grafana;

CREATE ROLE "modello-cnn3d" NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT LOGIN NOREPLICATION NOBYPASSRLS PASSWORD 'modello-cnn3d';
GRANT SELECT ON TABLE public.misura TO "modello-cnn3d";
GRANT INSERT ON TABLE public.previsione TO "modello-cnn3d";
GRANT SELECT ON TABLE public.previsione TO "modello-cnn3d";
GRANT INSERT ON TABLE public.step TO "modello-cnn3d";
GRANT SELECT ON TABLE public.modello TO "modello-cnn3d";